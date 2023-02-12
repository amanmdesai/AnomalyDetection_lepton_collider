import ROOT

# global parameters
intLumi        = 5.0e+06 #in pb-1
ana_tex        = 'e^{+}e^{-} #rightarrow Z H'
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
inputDir       = 'final/'
formats        = ['png','pdf']
yaxis          = ['lin','log']
stacksig       = ['stack','nostack']
outdir         = 'plots/'

variables =  [
            "jet_e",
            "jet_pt",
            "jet_phi",
            "jet_eta",

            "muon_e",
            "muon_pt",
            "muon_phi",
            "muon_eta",

            "electron_e",
            "electron_pt",
            "electron_phi",
            "electron_eta",
            ]


###Dictionary with the analysis name as a key, and the list of selections to be plotted for this analysis. The name of the selections should be the same than in the final selection
selections = {}
selections['sig']   = ["sel0"]#,"sel1"]

extralabel = {}
extralabel['sel0'] = ""

colors = {}
colors['sig'] = ROOT.kRed
colors['bgr'] = ROOT.kBlue

plots = {}
plots['sig'] = {'signal':{'sig':['signal']},
               'backgrounds':{'bgr':['background']}}

legend = {}
legend['sig'] = 'signal'
legend['bgr'] = 'background'
