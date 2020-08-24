#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

load("@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "auto_configure_fail",
    "auto_configure_warning",
    "get_starlark_list",
    "write_builtin_include_directory_paths",
)
load("@onedal//dev/bazel:utils.bzl",
    "unique",
)
load("@onedal//dev/bazel/toolchains:common.bzl",
    "TEST_CPP_FILE",
    "get_starlark_list_dict",
    "get_cxx_inc_directories",
    "add_compiler_option_if_supported",
    "add_linker_option_if_supported",
    "get_no_canonical_prefixes_opt",
    "get_toolchain_identifier",
    "get_default_compiler_options",
    "get_cpu_specific_options",
)

def _find_gold_linker_path(repo_ctx, cc):
    """Checks if `gold` is supported by the C compiler.
    Args:
      repo_ctx: repo_ctx.
      cc: path to the C compiler.
    Returns:
      String to put as value to -fuse-ld= flag, or None if gold couldn't be found.
    """
    result = repo_ctx.execute([
        cc,
        str(repo_ctx.path(TEST_CPP_FILE)),
        "-o",
        "/dev/null",
        # Some macos clang versions don't fail when setting -fuse-ld=gold, adding
        # these lines to force it to. This also means that we will not detect
        # gold when only a very old (year 2010 and older) is present.
        "-Wl,--start-lib",
        "-Wl,--end-lib",
        "-fuse-ld=gold",
        "-v",
    ])
    if result.return_code != 0:
        return None

    for line in result.stderr.splitlines():
        if line.find("gold") == -1:
            continue
        for flag in line.split(" "):
            if flag.find("gold") == -1:
                continue
            if flag.find("--enable-gold") > -1 or flag.find("--with-plugin-ld") > -1:
                # skip build configuration options of gcc itself
                continue

            # flag is '-fuse-ld=gold' for GCC or "/usr/lib/ld.gold" for Clang
            # strip space, single quote, and double quotes
            flag = flag.strip(" \"'")

            # remove -fuse-ld= from GCC output so we have only the flag value part
            flag = flag.replace("-fuse-ld=", "")
            return flag
    auto_configure_warning(
        "CC with -fuse-ld=gold returned 0, but its -v output " +
        "didn't contain 'gold', falling back to the default linker.",
    )
    return None

def _find_tool(repo_ctx, tool_name, mandatory=False):
    if tool_name.startswith("/"):
        return tool_name
    tool_path = repo_ctx.which(tool_name)
    if not tool_path:
        if mandatory:
            auto_configure_fail("Cannot find {}; try to correct your $PATH".format(tool_name))
        else:
            repo_ctx.template(
                "tool_not_found.sh",
                Label("@onedal//dev/bazel/toolchains/tools:tool_not_found.tpl.sh"),
                { "%{tool_name}": tool_name }
            )
            tool_path = repo_ctx.path("tool_not_found.sh")
    return str(tool_path)

def _create_ar_merge_tool(repo_ctx, ar_path):
    ar_merge_name = "merge_static_libs.sh"
    repo_ctx.template(
        ar_merge_name,
        Label("@onedal//dev/bazel/toolchains/tools:merge_static_libs_lnx.tpl.sh"),
        { "%{ar_path}": ar_path }
    )
    ar_merge_path = repo_ctx.path(ar_merge_name)
    return str(ar_merge_path)

def _find_tools(repo_ctx, reqs):
    # TODO: Use full compiler path from reqs
    ar = _find_tool(repo_ctx, "ar", mandatory=True)
    ar_merge = _create_ar_merge_tool(repo_ctx, ar)
    return struct(
        cc       = _find_tool(repo_ctx, reqs.compiler_id, mandatory=True),
        dpcc     = _find_tool(repo_ctx, reqs.dpc_compiler_id, mandatory=False),
        strip    = _find_tool(repo_ctx, "strip", mandatory=True),
        ar       = ar,
        ar_merge = ar_merge,
    )


def _preapre_builtin_include_directory_paths(repo_ctx, tools):
    builtin_include_directories = unique(
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc") +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++") +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "-xc",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "-xc++",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc) +
            _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc),
        )

    )
    write_builtin_include_directory_paths(repo_ctx, tools.cc, builtin_include_directories)
    return builtin_include_directories


def _get_bin_search_flag(repo_ctx, cc_path):
    cc_path = repo_ctx.path(cc_path)
    if not str(cc_path).startswith(str(repo_ctx.path(".")) + "/"):
        # cc is outside the repository, set -B
        bin_search_flag = ["-B" + str(cc_path.dirname)]
    else:
        # cc is inside the repository, don't set -B.
        bin_search_flag = []
    return bin_search_flag

def _get_gcc_toolchain_path(repo_ctx):
    return str(repo_ctx.which("gcc").dirname.dirname.realpath)

def _add_gcc_toolchain_if_needed(repo_ctx, cc):
    if ("clang" in cc) or ("dpcpp" in cc):
        return [ "--gcc-toolchain=" + _get_gcc_toolchain_path(repo_ctx) ]
    else:
        return []

def _add_gold_linker_path_if_available(repo_ctx, cc):
    gold_linker_path = _find_gold_linker_path(repo_ctx, cc)
    if gold_linker_path:
        return [ "-fuse-ld=" + gold_linker_path ]
    else:
        return []

def configure_cc_toolchain_lnx(repo_ctx, reqs):
    if reqs.os_id != "lnx":
        auto_configure_fail("Cannot configure Linux toolchain for '{}'".format(reqs.os_id))

    # Write empry C++ file to test compiler options
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    # Default compilations/link options
    cxx_opts  = [ ]
    link_opts = [ ]
    link_libs = [ "-lstdc++", "-lm", "-ldl" ]

    # Paths to tools/compiler includes
    tools = _find_tools(repo_ctx, reqs)
    builtin_include_directories = _preapre_builtin_include_directory_paths(repo_ctx, tools)

    # Addition compile/link flags
    bin_search_flag_cc = _get_bin_search_flag(repo_ctx, tools.cc)
    bin_search_flag_dpcc = _get_bin_search_flag(repo_ctx, tools.dpcc)

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_lnx.tpl.BUILD"),
        {
            # Various IDs
            "%{cc_toolchain_identifier}": get_toolchain_identifier(reqs),
            "%{compiler}":                reqs.compiler_id + "-" + reqs.compiler_version,
            "%{abi_version}":             reqs.compiler_abi_version,
            "%{abi_libc_version}":        reqs.libc_abi_version,
            "%{target_libc}":             reqs.libc_version,
            "%{target_cpu}":              reqs.target_arch_id,
            "%{host_system_name}":        reqs.os_id + "-" + reqs.host_arch_id,
            "%{target_system_name}":      reqs.os_id + "-" + reqs.target_arch_id,

            "%{supports_param_files}": "1",
            "%{cc_compiler_deps}": get_starlark_list(
                [":builtin_include_directory_paths"]
            ),

            # Tools
            "%{cc_path}":       tools.cc,
            "%{dpcc_path}":     tools.dpcc,
            "%{ar_path}":       tools.ar,
            "%{ar_merge_path}": tools.ar_merge,
            "%{strip_path}":    tools.strip,

            "%{cxx_builtin_include_directories}": get_starlark_list(builtin_include_directories),
            "%{compile_flags_cc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.cc,
                    is_dpcc = False,
                    category = "common",
                ) +
                add_compiler_option_if_supported(
                    # Option supported only by Intel Compiler to disable some warnings:
                    # https://software.intel.com/comment/1848937
                    repo_ctx,
                    tools.cc,
                    "-diag-disable=remark",
                )
            ),
            "%{compile_flags_dpcc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.dpcc,
                    is_dpcc = True,
                    category = "common",
                )
            ),
            "%{compile_flags_pedantic_cc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.cc,
                    is_dpcc = False,
                    category = "pedantic",
                )
            ),
            "%{compile_flags_pedantic_dpcc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.dpcc,
                    is_dpcc = True,
                    category = "pedantic",
                )
            ),
            "%{cxx_flags}": get_starlark_list(cxx_opts),
            "%{link_flags_cc}": get_starlark_list(
                _add_gold_linker_path_if_available(repo_ctx, tools.cc) +
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,-no-as-needed",
                    "-no-as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.cc,
                    "-pass-exit-codes",
                ) +
                (
                    [ "-no-cilk", "-static-intel", ] if reqs.compiler_id == "icc" else []
                ) +
                bin_search_flag_cc + link_opts
            ),
            "%{link_flags_dpcc}": get_starlark_list(
                _add_gold_linker_path_if_available(repo_ctx, tools.dpcc) +
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-no-as-needed",
                    "-no-as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.dpcc,
                    "-pass-exit-codes",
                ) +
                bin_search_flag_dpcc + link_opts
            ),
            "%{link_libs}": get_starlark_list(link_libs),
            "%{opt_compile_flags}": get_starlark_list(
                [
                    # No debug symbols.
                    # Maybe we should enable https://gcc.gnu.org/wiki/DebugFission for opt or
                    # even generally? However, that can't happen here, as it requires special
                    # handling in Bazel.
                    "-g0",

                    # Conservative choice for -O
                    # -O3 can increase binary size and even slow down the resulting binaries.
                    # Profile first and / or use FDO if you need better performance than this.
                    "-O2",

                    # Security hardening on by default.
                    # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
                    "-D_FORTIFY_SOURCE=1",

                    # Disable assertions
                    "-DNDEBUG",

                    # Removal of unused code and data at link time (can this increase binary
                    # size in some cases?).
                    "-ffunction-sections",
                    "-fdata-sections",
                ],
            ),
            "%{opt_link_flags}": get_starlark_list(
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,--gc-sections",
                    "-gc-sections",
                ),
            ),
            "%{no_canonical_system_headers_flags_cc}": get_starlark_list(
                # Probably bug: Intel Compiler links OpenMP runtime if
                # -fno-canonical-system-headers is provided
                (get_no_canonical_prefixes_opt(repo_ctx, tools.cc)
                 if reqs.compiler_id != "icc" else [])
            ),
            "%{no_canonical_system_headers_flags_dpcc}": get_starlark_list(
                get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc)
            ),
            "%{deterministic_compile_flags}": get_starlark_list(
                [
                    # Make C++ compilation deterministic. Use linkstamping instead of these
                    # compiler symbols.
                    "-Wno-builtin-macro-redefined",
                    "-D__DATE__=\\\"redacted\\\"",
                    "-D__TIMESTAMP__=\\\"redacted\\\"",
                    "-D__TIME__=\\\"redacted\\\"",
                ],
            ),
            "%{dbg_compile_flags}": get_starlark_list(
                [
                    "-g",

                    # Disable optimizations explicitly
                    # Some compilers like Intel uses -O2 by default
                    "-O0",
                ]
            ),
            "%{supports_start_end_lib}": "False" if reqs.compiler_id == "icc" else "True",
            "%{supports_random_seed}": "False" if reqs.compiler_id == "icc" else "True",
            "%{cpu_flags}": get_starlark_list_dict(
                get_cpu_specific_options(reqs)
            )
        },
    )
