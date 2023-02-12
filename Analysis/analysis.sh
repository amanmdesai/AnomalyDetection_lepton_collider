export FCCDICTSDIR=/
fccanalysis run analysis_stage1.py --output signal.root --files-list ../Samples/signal.root
fccanalysis run analysis_stage1.py --output background.root --files-list ../Samples/background.root
fccanalysis final analysis_final.py
fccanalysis plots analysis_plots.py



#python uproot_covert.py signal
#python uproot_covert.py background
