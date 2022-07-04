#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

PYTHON_VERSION=$1
RELEASE_VERSION=$2  # rX.Y or nightly
DEFAULT_PYTHON_VERSION=3.8
DEBIAN_FRONTEND=noninteractive

ACL_VERSION=v22.05
ONEDNN_VERSION=v2.6

mkdir $HOME/torch_xla_wheel_builder
WHEEL_BUILDER_DIR=$HOME/torch_xla_wheel_builder

function maybe_append {
  local LINE="$1"
  local FILE="$2"
  if [ ! -r "$FILE" ]; then
    if [ -w "$(dirname $FILE)" ]; then
      echo "$LINE" > "$FILE"
    else
      sudo bash -c "echo '$LINE' > \"$FILE\""
    fi
  elif [ "$(grep -F "$LINE" $FILE)" == "" ]; then
    if [ -w "$FILE" ]; then
      echo "$LINE" >> "$FILE"
    else
      sudo bash -c "echo '$LINE' >> \"$FILE\""
    fi
  fi
}

function setup_system {
  maybe_append 'APT::Acquire::Retries "10";' /etc/apt/apt.conf.d/80-failparams
  maybe_append 'APT::Acquire::http::Timeout "180";' /etc/apt/apt.conf.d/80-failparams
  maybe_append 'APT::Acquire::ftp::Timeout "180";' /etc/apt/apt.conf.d/80-failparams
}

function maybe_install_sources {
  cd $WHEEL_BUILDER_DIR
  if [ ! -d "torch" ]; then
    sudo apt-get install -y git
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    if [ "${RELEASE_VERSION}" != "nightly" ]; then
       git checkout $RELEASE_VERSION -b $RELEASE_VERSION
    fi
    git clone --recursive https://github.com/snadampal/xla.git
    cd xla
    git checkout torch_xla_tf_update
  fi
}

function install_bazel() {
 cd $WHEEL_BUILDER_DIR
 version=5.1.1
 if [ ! -d "bazel" ]; then
   mkdir $WHEEL_BUILDER_DIR/bazel
   wget https://github.com/bazelbuild/bazel/releases/download/$version/bazel-$version-linux-arm64 -O $WHEEL_BUILDER_DIR/bazel/bazel
 fi
 chmod a+x $WHEEL_BUILDER_DIR/bazel/bazel
 export PATH=$WHEEL_BUILDER_DIR/bazel:$PATH
}

function install_ninja() {
  sudo apt-get install ninja-build
}

function debian_version {
  local VER
  if ! sudo apt-get install -y lsb-release > /dev/null 2>&1 ; then
    VER="buster"
  else
    VER=$(lsb_release -c -s)
  fi
  echo "$VER"
}

function install_req_packages() {
  sudo apt-get -y install python3-pip git curl libopenblas-dev vim apt-transport-https ca-certificates wget procps libssl-dev scons gcc-10 g++-10
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
  sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
  export CC=/usr/bin/gcc-10 export CXX=/usr/bin/g++-10
  pip install --no-cache-dir typing_extensions numpy lark-parser
  install_bazel
  install_ninja
}

function install_latest_cmake() {
  # install the version >=3.18
  version=3.22
  build=2
  mkdir $WHEEL_BUILDER_DIR/temp
  cd $WHEEL_BUILDER_DIR/temp
  wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
  tar -xzvf cmake-$version.$build.tar.gz
  cd cmake-$version.$build/
  ./bootstrap
  make -j$(nproc)
  sudo make install
  # check the version
  cmake --version
}

function build_acl() {
  readonly version=$ACL_VERSION
  readonly src_host=https://review.mlplatform.org/ml
  readonly src_repo=ComputeLibrary

  mkdir $WHEEL_BUILDER_DIR/acl
  install_dir=$WHEEL_BUILDER_DIR/acl

  git clone ${src_host}/${src_repo}.git
  cd ${src_repo}
  git checkout $version

  # Build with scons
  scons -j16  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
     os=linux arch=armv8.2-a build=native multi_isa=1 \
     build_dir=$install_dir/build

  cp -r arm_compute $install_dir
  cp -r include $install_dir
  cp -r utils $install_dir
  cp -r support $install_dir
}

function build_and_install_torch() {
  cd $WHEEL_BUILDER_DIR/pytorch
  # Checkout the PT commit ID or branch if we have one.
  COMMITID_FILE="xla/.torch_pin"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  # Only checkout dependencies once PT commit/branch checked out.
  git submodule sync
  git submodule update --init --recursive

  cd third_party/ideep/mkl-dnn/third_party/oneDNN
  git checkout $ONEDNN_VERSION

  cd $WHEEL_BUILDER_DIR/pytorch
  # Apply patches to PT which are required by the XLA support.
  xla/scripts/apply_patches.sh
  export ACL_ROOT_DIR=$WHEEL_BUILDER_DIR/acl/
  BLAS="OpenBLAS" USE_OPENMP=1 USE_MKLDNN=ON USE_MKLDNN_ACL=ON python setup.py bdist_wheel
  pip install dist/*.whl
}

function build_and_install_torch_xla() {
  cd $WHEEL_BUILDER_DIR/pytorch/xla
  export PATH=$WHEEL_BUILDER_DIR/bazel:$PATH
  # enable ACL runtime for CPU
  export XLA_CPU_USE_ACL=1
  export BUILD_CPP_TESTS=0

  git submodule update --init --recursive
  if [ "${RELEASE_VERSION}" = "nightly" ]; then
    export VERSIONED_XLA_BUILD=1
  else
    export TORCH_XLA_VERSION=${RELEASE_VERSION:1}  # r0.5 -> 0.5
  fi

  patch -p1 < ./torch_xla.patch

  cd third_party/tensorflow
  patch -p1 < ../../xla_cpu_enhancements.patch

  cd ../../
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function install_torchvision_from_source() {
  cd $WHEEL_BUILDER_DIR
  torchvision_repo_version="main"
  # Cannot install torchvision package with PyTorch installation from source.
  # https://github.com/pytorch/vision/issues/967
  git clone -b "${torchvision_repo_version}" https://github.com/pytorch/vision.git
  pushd vision
  python setup.py bdist_wheel
  pip install dist/*.whl
  popd
}

function main() {
  setup_system
  install_req_packages
  install_latest_cmake
  build_acl
  maybe_install_sources
  build_and_install_torch
  build_and_install_torch_xla
  install_torchvision_from_source
}

main
