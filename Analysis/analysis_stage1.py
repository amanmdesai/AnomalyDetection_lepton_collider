inputDir  = "Samples"
outputDir = "stage1"

nCPUS     = 8


class RDFanalysis():

    def analysers(df):
        df2 = (
            df
            # First using the jets as defined in the EDM4HEP file
            #.Alias("Jet3","Jet#3.index")
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


            .Alias("Muon0", "Muon#0.index")
            .Define("muons",  "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
            .Define("muon_e", "ReconstructedParticle::get_e(muons)")
            .Define("muon_pt", "ReconstructedParticle::get_pt(muons)")
            .Define("muon_eta", "ReconstructedParticle::get_eta(muons)")
            .Define("muon_phi", "ReconstructedParticle::get_phi(muons)")


            .Alias("Electron0", "Electron#0.index")
            .Define("electrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
            .Define("electron_e", "ReconstructedParticle::get_e(electrons)")
            .Define("electron_pt", "ReconstructedParticle::get_pt(electrons)")
            .Define("electron_eta", "ReconstructedParticle::get_eta(electrons)")
            .Define("electron_phi", "ReconstructedParticle::get_phi(electrons)")

        )
        return df2


    def output():
        branchList = [

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
        return branchList
