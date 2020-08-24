#!/bin/bash
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --alg-name)
        ALG_NAME="$2"
        ;;
        --python-version)
        PYTHON_VERSION="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done
SKLEARN_PATH="$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/sklearn/"

case ${ALG_NAME} in
    "dbscan")
        cp ${SKLEARN_PATH}cluster/tests/test_dbscan.py test_dbscan.py
    ;;
    "elastic_net")
        cp ${SKLEARN_PATH}linear_model/tests/test_coordinate_descent.py test_elastic_net.py
    ;;
    "kmeans")
        cp ${SKLEARN_PATH}cluster/tests/test_k_means.py test_kmeans.py
    ;;
    "lin_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_base.py test_lin_reg.py
    ;;
    "log_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_logistic.py test_log_reg.py
    ;;
    "pca")
        cp ${SKLEARN_PATH}decomposition/tests/test_pca.py test_pca.py
    ;;
    "ridge_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_ridge.py test_ridge_reg.py
    ;;
    "svm")
        cp ${SKLEARN_PATH}svm/tests/test_svm.py test_svm.py
    ;;
    "svm_sparse")
        cp ${SKLEARN_PATH}svm/tests/test_sparse.py test_svm_sparse.py
    ;;
    *)
        echo "Unknown algorithm: ${ALG_NAME}"
        exit 1
    ;;
esac
