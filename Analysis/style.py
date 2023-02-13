import ROOT
def MyStyle(font  = 22,tsize = 17):
	MyStyle = ROOT.TStyle("MyStyle","MyStyle")
	MyStyle.SetPaperSize(25,25)
	MyStyle.SetPadTopMargin(0.10)
	MyStyle.SetPadRightMargin(0.10)
	MyStyle.SetPadBottomMargin(0.10)
	MyStyle.SetPadLeftMargin(0.10)
	MyStyle.SetFrameBorderMode(0)
	MyStyle.SetFrameFillColor(0)
	MyStyle.SetPadColor(0)
	MyStyle.SetTextFont(font)
	MyStyle.SetTextSize(0.03)
	MyStyle.SetTitleXOffset(1.)
	MyStyle.SetTitleYOffset(2.5)
	MyStyle.SetNdivisions(510,"x")
	MyStyle.SetNdivisions(505,"y")
	ROOT.gROOT.SetStyle("MyStyle");
	ROOT.gROOT.ForceStyle();
