srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash
module load devel/CMake devel/SQLite/ compiler/NVHPC
export LD_LIBRARY_PATH=/opt/apps/resif/iris-rhel8/2020b/gpu/software/NVHPC/21.2/Linux_x86_64/21.2/math_libs/11.2/lib64:$LD_LIBRARY_PATH