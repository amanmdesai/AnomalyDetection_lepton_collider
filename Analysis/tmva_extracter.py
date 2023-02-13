import ROOT

from ROOT import TMVA,gPad,TCanvas,TH1,TFile,TFrame,TLegend,gFile,TDirectory,TColor
from style import MyStyle

MyStyle()

filename = "results_tmva.root"


methodname =["BDT"]# ["Cuts","Likelihood","BDT","BDT-Grad","HMatrix","CFMlpANN","KNN","Fisher","MLP","DL1r-try"]
var = ["0.80"]#[", 0.75",", 0.75",", 0.75",", 0.75",", 0.75",", 0.75",", 0.75",", 0.75",", 0.75",", 0.75"]
f = TFile.Open(filename,"read")
c = ROOT.TCanvas()
legend = ROOT.TLegend(0.18,0.45,0.45,0.86)
#legend.SetHeader("       ML Method")
#https://root-forum.cern.ch/t/tmva-roc-curve-extraction-and-modification/29395
i=0
for method in methodname:
	i=i+1

	if "DL1" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_PyKeras/"+method)
		objectName = "MVA_"+method+"_effBvsS"
		method = "DeepLearning"

	elif "Grad" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_BDT/"+method)
		objectName = "MVA_"+method+"_effBvsS"
	else:
		dFolder  =   gFile.GetDirectory("/dataset/Method_"+method+"/"+method)
		objectName = "MVA_"+method+"_effBvsS"


	h = dFolder.Get(objectName)


	if i==10:
		h.SetLineColor(ROOT.kMagenta+3)
	else:
		h.SetLineColor(i)
	h.SetLineStyle(1)
	h.Draw("same")
	legend.AddEntry(h,method,"L")
	h.GetXaxis().SetTitle("Signal Efficiency")
	h.GetYaxis().SetTitle("Background Efficiency #epsilon_{b}")
	h.SetMaximum(1.)
	h.SetMinimum(0.)
	#h.GetXaxis().SetRange(40,60)
legend.SetBorderSize(0)
legend.SetBorderSize(0)
legend.SetFillColor(0)
legend.SetTextSize(0.037)
legend.SetTextFont(42)
legend.Draw()
c.SaveAs("dnn/roc_effsvseffs.pdf")




c2 = ROOT.TCanvas()
legend = ROOT.TLegend(0.18,0.2,0.45,0.6)
#legend.SetHeader("       ML Method, ROC Integral")
#https://root-forum.cern.ch/t/tmva-roc-curve-extraction-and-modification/29395
i=0
for method in methodname:
	i=i+1

	if "DL1" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_PyKeras/"+method)
		objectName = "MVA_"+method+"_rejBvsS"
		method = "DeepLearning"

	elif "Grad" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_BDT/"+method)
		objectName = "MVA_"+method+"_rejBvsS"
	else:
		dFolder  =   gFile.GetDirectory("/dataset/Method_"+method+"/"+method)
		objectName = "MVA_"+method+"_rejBvsS"


	h = dFolder.Get(objectName)


	if i==10:
		h.SetLineColor(ROOT.kMagenta+3)
	else:
		h.SetLineColor(i)
	h.SetLineStyle(1)
	h.Draw("same")
	legend.AddEntry(h,method+var[i-1],"L")
	h.GetXaxis().SetTitle("Signal Efficiency")
	h.GetYaxis().SetTitle("Background Rejection (1- #epsilon_{b})")
	h.SetMaximum(1.)
	h.SetMinimum(0.)
legend.SetBorderSize(0)
legend.SetBorderSize(0)
legend.SetFillColor(0)
legend.SetTextSize(0.037)
legend.SetTextFont(42)
legend.Draw()
c2.SaveAs("dnn/methods_roc_rejvsEffs.pdf")



c3 = ROOT.TCanvas()
legend = ROOT.TLegend(0.7,0.8,0.9,0.9)
#legend.SetHeader("       ML Method")
#https://root-forum.cern.ch/t/tmva-roc-curve-extraction-and-modification/29395
i=0
for method in methodname:
	i=i+1
	if "Cuts" in method:
		continue
	if "DL1" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_PyKeras/"+method)
		objectName = "MVA_"+method+"_invBeffvsSeff"
		method = "DeepLearning"

	elif "Grad" in method:
		dFolder  =   gFile.GetDirectory("/dataset/Method_BDT/"+method)
		objectName = "MVA_"+method+"_invBeffvsSeff"
	else:
		dFolder  =   gFile.GetDirectory("/dataset/Method_"+method+"/"+method)
		objectName = "MVA_"+method+"_invBeffvsSeff"


	h = dFolder.Get(objectName)


	if i==10:
		h.SetLineColor(ROOT.kMagenta+3)
	else:
		h.SetLineColor(i)
	h.SetLineStyle(1)
	h.Draw("same")
	legend.AddEntry(h,method+" AUC "+var[0],"L")
	h.GetXaxis().SetTitle("Signal Efficiency")
	h.GetYaxis().SetTitle("Background Rejection (1/#epsilon_{b})")
	h.SetMaximum(h.GetMaximum()*50)
	h.SetMinimum(1)
	#h.GetXaxis().SetRange(40,60)
legend.SetBorderSize(0)
legend.SetBorderSize(0)
legend.SetFillColor(0)
legend.SetTextSize(0.037)
legend.SetTextFont(42)
legend.Draw()
c3.SetLogy()
c3.SaveAs("dnn/methods_roc_InvBEffvsSEffs.pdf")
