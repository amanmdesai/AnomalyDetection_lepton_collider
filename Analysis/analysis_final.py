processList = {
    'signal':{},
    'background':{},
}

inputDir  = "stage1"
outputDir = "final"


nCPUS     = 8

#Link to the dictonary that contains all the cross section informations etc...
procDict = "/FCC/GenScripts/FCCee_procDict_spring2021_IDEA.json"

procDictAdd={
    "signal":{"numberOfEvents": 10000, "sumOfWeights": 10000, "crossSection": 0.1, "kfactor": 1.0, "matchingEfficiency": 1.0},
    "background":{"numberOfEvents": 10000, "sumOfWeights": 10000, "crossSection": 0.05, "kfactor": 1.0, "matchingEfficiency": 1.0}
}
#produces ROOT TTrees, default is False
doTree = True

###Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
cutList = {"sel0":"jet_e.size()>=0."}
#Dictionary for the ouput variable/hitograms. The key is the name of the variable in the output files. "name" is the name of the variable in the input file, "title" is the x-axis label of the histogram, "bin" the number of bins of the histogram, "xmin" the minimum x-axis value and "xmax" the maximum x-axis value.
histoList = {

    "jet_e":{"name":"jet_e","title":"Jet Energy [GeV]","bin":25,"xmin":0,"xmax":160},
    "muon_e":{"name":"muon_e","title":"Muon Energy [GeV]","bin":25,"xmin":0,"xmax":160},
    "electron_e":{"name":"electron_e","title":"Electron Energy [GeV]","bin":25,"xmin":0,"xmax":160},

    "jet_pt":{"name":"jet_pt","title":"Jet pT [GeV]","bin":25,"xmin":0,"xmax":160},
    "muon_pt":{"name":"muon_pt","title":"Muon pT [GeV]","bin":25,"xmin":0,"xmax":160},
    "electron_pt":{"name":"electron_pt","title":"Electron pT [GeV]","bin":25,"xmin":0,"xmax":160},

    "jet_phi":{"name":"jet_phi","title":"Jet #phi","bin":15,"xmin":-6,"xmax":6},
    "muon_phi":{"name":"muon_phi","title":"Muon #phi","bin":15,"xmin":-6,"xmax":6},
    "electron_phi":{"name":"electron_phi","title":"Electron #phi","bin":15,"xmin":-6,"xmax":6},

    "jet_eta":{"name":"jet_eta","title":"Jet #eta","bin":15,"xmin":-6,"xmax":6},
    "muon_eta":{"name":"muon_eta","title":"Muon #eta","bin":15,"xmin":-6,"xmax":6},
    "electron_eta":{"name":"electron_eta","title":"Electron #eta","bin":15,"xmin":-6,"xmax":6},


}
