workspace(name = "onedal")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "bazel_skylib",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
  ],
  sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)

load("@onedal//dev/bazel/config:config.bzl", "declare_onedal_config")
declare_onedal_config(
    name = "config",
)

load("@onedal//dev/bazel/toolchains:cc_toolchain.bzl", "declare_onedal_cc_toolchain")
declare_onedal_cc_toolchain(
    name = "onedal_cc_toolchain",
)

load("@onedal//dev/bazel/toolchains:extra_toolchain.bzl", "declare_onedal_extra_toolchain")
declare_onedal_extra_toolchain(
    name = "onedal_extra_toolchain",
)

load("@onedal//dev/bazel/deps:micromkl.bzl", "micromkl_repo", "micromkl_dpc_repo")
micromkl_repo(
    name = "micromkl",
    root_env_var = "MKLFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklfpk_lnx_2021.1-beta08.tgz",
    sha256 = "cc1142f0cfd831e394a09231f89946ce84e87b33212845a46d5e869370570962",
)

micromkl_dpc_repo(
    name = "micromkl_dpc",
    root_env_var = "MKLGPUFPKROOT",
    url = "https://github.com/oneapi-src/oneDAL/releases/download/Dependencies/mklgpufpk_lnx_20200414.tgz",
    sha256 = "24ba3bb79693d691632da505378d975f903e830568fe2b6ce671156f4c9d568f",
)

load("@onedal//dev/bazel/deps:tbb.bzl", "tbb_repo")
tbb_repo(
    name = "tbb",
    root_env_var = "TBBROOT",
    url = "https://github.com/oneapi-src/oneTBB/releases/download/v2021.1-beta08/oneapi-tbb-2021.1-beta08-lin.tgz",
    sha256 = "02cfd300e3880f2376457feeb8d29e0a3aabf2ac4caf0883b57ca1a2dba073f0",
    strip_prefix = "oneapi-tbb-2021.1-beta08",
)

load("@onedal//dev/bazel/deps:onedal.bzl", "onedal_repo")
onedal_repo(
    name = "onedal_release",
    root_env_var = "DAALROOT",
)

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
)
