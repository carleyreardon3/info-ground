DEVICE=$1

if [ "$DEVICE" == "cpu" ]; then
    condor_submit -i htc_scripts/interactive_cpu.sub
elif [ "$DEVICE" == "gpu" ]; then
    condor_submit -i htc_scripts/interactive_gpu.sub
else
    echo "DEVICE must be one of {'cpu','gpu'}"
fi