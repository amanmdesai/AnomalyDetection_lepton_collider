inputDir  = "Samples"
outputDir = "stage1"

nCPUS     = 8


class RDFanalysis():

    def analysers(df):
        df2 = (
            df
            ## get information about reconstructed particles
            .Define("RP_px",          "ReconstructedParticle::get_px(ReconstructedParticles)")
            .Define("RP_py",          "ReconstructedParticle::get_py(ReconstructedParticles)")
            .Define("RP_pz",          "ReconstructedParticle::get_pz(ReconstructedParticles)")
            .Define("RP_phi",          "ReconstructedParticle::get_phi(ReconstructedParticles)")
            .Define("RP_e",           "ReconstructedParticle::get_e(ReconstructedParticles)")
            .Define("RP_m",           "ReconstructedParticle::get_mass(ReconstructedParticles)")
            .Define("RP_q",           "ReconstructedParticle::get_charge(ReconstructedParticles)")


            .Define("pseudo_jets",    "JetClusteringUtils::set_pseudoJets(RP_px, RP_py, RP_pz, RP_e)")


            .Define("FCCAnalysesJets_eekt", "JetClustering::clustering_ee_kt(2, 2, 1, 0)(pseudo_jets)")
            .Define("jets_eekt",           "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_eekt)")


            .Define("jet_e",        "JetClusteringUtils::get_e(jets_eekt)")
            .Define("jet_pt",        "JetClusteringUtils::get_pt(jets_eekt)")
            .Define("jet_eta",        "JetClusteringUtils::get_eta(jets_eekt)")
            .Define("jet_phi",        "JetClusteringUtils::get_phi(jets_eekt)")

            .Define("n_jet",        "return jets_eekt.size()")


            .Define("jet_e_1", "if(n_jet > 0){return 1.0*jet_e[0];} else return -99.;")
            .Define("jet_pt_1", "if(n_jet > 0){return 1.0*jet_pt[0];} else return -99.;")
            .Define("jet_eta_1", "if(n_jet > 0){return 1.0*jet_eta[0];} else return -99.;")
            .Define("jet_phi_1", "if(n_jet > 0){return 1.0*jet_phi[0];} else return -99.;")

            .Define("jet_e_2", "if(n_jet > 1){return 1.0*jet_e[1];} else return -99.;")
            .Define("jet_pt_2", "if(n_jet > 1){return 1.0*jet_pt[1];} else return -99.;")
            .Define("jet_eta_2", "if(n_jet > 1){return 1.0*jet_eta[1];} else return -99.;")
            .Define("jet_phi_2", "if(n_jet > 1){return 1.0*jet_phi[1];} else return -99.;")


            .Alias("Muon0", "Muon#0.index")
            .Define("muons",  "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
            .Define("n_muon",        "return muons.size()")

            .Define("muon_e", "ReconstructedParticle::get_e(muons)")
            .Define("muon_pt", "ReconstructedParticle::get_pt(muons)")
            .Define("muon_eta", "ReconstructedParticle::get_eta(muons)")
            .Define("muon_phi", "ReconstructedParticle::get_phi(muons)")

            .Define("muon_e_1", "if(n_muon > 0){return 1.0*muon_e[0];} else return -99.;")
            .Define("muon_pt_1", "if(n_muon > 0){return 1.0*muon_pt[0];} else return -99.;")
            .Define("muon_eta_1", "if(n_muon > 0){return 1.0*muon_eta[0];} else return -99.;")
            .Define("muon_phi_1", "if(n_muon > 0){return 1.0*muon_phi[0];} else return -99.;")

            .Define("muon_e_2", "if(n_muon > 1){return 1.0*muon_e[1];} else return -99.;")
            .Define("muon_pt_2", "if(n_muon > 1){return 1.0*muon_pt[1];} else return -99.;")
            .Define("muon_eta_2", "if(n_muon > 1){return 1.0*muon_eta[1];} else return -99.;")
            .Define("muon_phi_2", "if(n_muon > 1){return 1.0*muon_phi[1];} else return -99.;")


            .Alias("Electron0", "Electron#0.index")
            .Define("electrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
            .Define("n_electron",        "return electrons.size()")

            .Define("electron_e", "ReconstructedParticle::get_e(electrons)")
            .Define("electron_pt", "ReconstructedParticle::get_pt(electrons)")
            .Define("electron_eta", "ReconstructedParticle::get_eta(electrons)")
            .Define("electron_phi", "ReconstructedParticle::get_phi(electrons)")

            .Define("electron_e_1", "if(n_electron > 0){return 1.0*electron_e[0];} else return -99.;")
            .Define("electron_pt_1", "if(n_electron > 0){return 1.0*electron_pt[0];} else return -99.;")
            .Define("electron_eta_1", "if(n_electron > 0){return 1.0*electron_eta[0];} else return -99.;")
            .Define("electron_phi_1", "if(n_electron > 0){return 1.0*electron_phi[0];} else return -99.;")

            .Define("electron_e_2", "if(n_electron > 1){return 1.0*electron_e[1];} else return -99.;")
            .Define("electron_pt_2", "if(n_electron > 1){return 1.0*electron_pt[1];} else return -99.;")
            .Define("electron_eta_2", "if(n_electron > 1){return 1.0*electron_eta[1];} else return -99.;")
            .Define("electron_phi_2", "if(n_electron > 1){return 1.0*electron_phi[1];} else return -99.;")


        )
        return df2


    def output():
        branchList = [

            "n_jet",
            "n_muon",
            "n_electron",

            "jet_e",
            "jet_pt",
            "jet_phi",
            "jet_eta",

            "jet_e_1",
            "jet_pt_1",
            "jet_phi_1",
            "jet_eta_1",

            "jet_e_2",
            "jet_pt_2",
            "jet_phi_2",
            "jet_eta_2",

            "muon_e",
            "muon_pt",
            "muon_phi",
            "muon_eta",

            "muon_e_1",
            "muon_pt_1",
            "muon_phi_1",
            "muon_eta_1",

            "muon_e_2",
            "muon_pt_2",
            "muon_phi_2",
            "muon_eta_2",

            "electron_e",
            "electron_pt",
            "electron_phi",
            "electron_eta",

            "electron_e_1",
            "electron_pt_1",
            "electron_phi_1",
            "electron_eta_1",

            "electron_e_2",
            "electron_pt_2",
            "electron_phi_2",
            "electron_eta_2",


        ]
        return branchList
