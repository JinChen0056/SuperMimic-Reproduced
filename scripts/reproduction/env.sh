# Source from the repository root before local training or tests.
unset PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
# Ubuntu's runtime-only Python may lack headers needed by Triton. Use a local
# extraction of libpython3.12-dev; no sudo or system package changes required.
sm_header_root="$PWD/data/toolchain/python312/usr/include"
if [[ -f "$sm_header_root/python3.12/Python.h" ]]; then
  export CPATH="$sm_header_root/python3.12:$sm_header_root${CPATH:+:$CPATH}"
fi
unset sm_header_root
