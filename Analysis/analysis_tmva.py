import ROOT

output = ROOT.TFile.Open('results_tmva.root', 'RECREATE')

data_sig =ROOT.TFile.Open('stage1/signal.root','read')
data_bgr =ROOT.TFile.Open('stage1/background.root','read')


signal = data_sig.Get('events')
background = data_bgr.Get('events')

dataloader = ROOT.TMVA.DataLoader('dataset')
factory = ROOT.TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=Norm:AnalysisType=Classification')

for branch in signal.GetListOfBranches():
    if "1" in branch.GetName():
        print(branch.GetName())
        dataloader.AddVariable(branch.GetName())
    elif "2" in branch.GetName():
        print(branch.GetName())
        dataloader.AddVariable(branch.GetName())
    else:
        continue
dataloader.AddSignalTree(signal, 2.0) #weight_mc
dataloader.AddBackgroundTree(background, 1.0)

selcut ='!TMath::IsNaN(n_jet)'

dataloader.PrepareTrainingAndTestTree(ROOT.TCut(selcut),"nTrain_Signal=7000:nTrain_Background=7000:nTest_Signal=3000:nTest_Background=3000:SplitMode=Random:!V:NormMode=NumEvents:MixMode=Random" );


factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT",'!H:!V:VarTransform=Norm:NTrees=100:MinNodeSize=3%:MaxDepth=5:BoostType=AdaBoost:SeparationType=GiniIndex')#:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=2:Shrinkage=0.10')
#factory.BookMethod( dataloader, ROOT.TMVA.Types.kMLP, "MLP", '!H:!V:NeuronType=sigmoid:VarTransform=Norm:NCycles=10:HiddenLayers=1')
#factory.BookMethod( dataloader, ROOT.TMVA.Types.kFisher, "Fisher", "H:!V:Fisher:VarTransform=Norm") #:CreateMVAPdfs:PDFInterpolMVAPdf=Spline2:NbinsMVAPdf=50:NsmoothMVAPdf=10" );
#factory.BookMethod(dataloader,  ROOT.TMVA.Types.kCuts,'Cuts','!H:!V:FitMethod=MC:EffSel:SampleSize=10000:VarProp=FSmart:VarTransform=Norm')
#factory.BookMethod(dataloader,  ROOT.TMVA.Types.kLikelihood, "Likelihood",'!H:!V:TransformOutput:PDFInterpol=Spline2:NAvEvtPerBin=25')


# In[7]:


factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
