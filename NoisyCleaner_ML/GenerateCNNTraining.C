/*
 * GenerateCNNTraining.C
 * Generates data specifically designed for CNN training.
 * * OUTPUT STRUCTURE (TTree 'training_data'):
 * 1. spectrum (Input): The noisy data with background.
 * 2. clean_signal (Target): The perfect peaks (no noise/bg). Great for U-Net targets.
 * 3. peak_centroids (Label): Vector of true positions.
 * 4. peak_amplitudes (Label): Vector of true heights.
 * 5. peak_sigmas     (Label): Vector of true widths.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TF1.h"
#include "TRandom3.h"
#include "TMath.h"

// --- CONFIGURATION ---
const int N_BINS = 1024;       // 4096 is ideal for CNN pooling layers (2^12)
const double MAX_ENERGY = 1024.0;
const int N_EVENTS = 50000;    // Size of dataset

// Resolution Function: Sigma = a + b * sqrt(E)
double GetSigma(double energy) {
    return 1.0 + 0.05 * std::sqrt(energy);
}

// Helper: Calculate Compton Edge Energy
double GetComptonEdge(double E_gamma) {
    // E_edge = E / (1 + m_e*c^2 / (2*E))
    // m_e*c^2 approx 511 keV
    double electron_mass = 511.0; 
    return E_gamma * ( (2*E_gamma/electron_mass) / (1 + 2*E_gamma/electron_mass) );
}

// Function to add a "Physics Realistic" peak to a histogram
void AddPhysicsPeak(TH1D* hist, TH1D* signalHist, double E_centroid, double amplitude, TF1* gauss, TF1* compton) {

    // 1. Setup Parameters
    double sigma = GetSigma(E_centroid); // Use your existing resolution function
    double fwhm = 2.355 * sigma;
    TRandom3 rnd(0);

    // --- A. THE PHOTOPEAK (Main Signal) ---
    gauss->SetParameters(amplitude, E_centroid, sigma);
    
    // We add this to BOTH the "Noisy Input" (hist) and the "Clean Target" (signalHist)
    // Because the U-Net should find this!
    int counts = (int)(amplitude * sigma * sqrt(2*TMath::Pi()));
    hist->FillRandom("g", counts); 
    
    // For the "Clean Signal", we usually fill weights directly for a perfect shape
    for(int b=1; b<=signalHist->GetNbinsX(); ++b) {
        double x = signalHist->GetBinCenter(b);
        double val = amplitude * TMath::Exp(-0.5 * pow((x-E_centroid)/sigma, 2));
        signalHist->SetBinContent(b, signalHist->GetBinContent(b) + val);
    }

    // --- B. THE COMPTON CONTINUUM (Background Noise) ---
    // This goes into 'hist' (Input) but NOT 'signalHist' (Target).
    // The U-Net must learn to REMOVE this.
    
    // We approximate the continuum as a Step Function (Erfc) that drops off at the Compton Edge
    double E_edge = GetComptonEdge(E_centroid);
    
    // The height of the Compton shelf relative to the Peak.
    // In real detectors, P/T (Peak-to-Total) ratio varies. Let's say Compton is ~10-30% of peak height.
    double compton_height = amplitude * rnd.Uniform(0.1, 0.3);
    
    // Create a smeared step function: 0.5 * Height * Erfc((x - Edge) / width)
    // We use a broader width for the edge smearing
    compton->SetParameters(compton_height, E_edge, sigma * 2.0); 
    
    // Limit the fill to below the peak to save time
    compton->SetRange(0, E_centroid); 
    
    // We add this ONLY to the input histogram
    hist->FillRandom("compton", (int)(counts * 0.5)); // Add proportional counts
    
    
    // --- C. ESCAPE PEAKS (Background Artifacts) ---
    // These are also "Noise" for the U-Net (usually), or separate classes.
    // Let's treat them as Noise (U-Net should remove them to find the true gamma energy).
    
    if (E_centroid > 1022.0) {
        double E_se = E_centroid - 511.0;  // Single Escape
        double E_de = E_centroid - 1022.0; // Double Escape
        
        // Probability depends on detector size. Let's approximate small peaks.
        double amp_se = amplitude * 0.15;
        double amp_de = amplitude * 0.08;
        
        // Add Single Escape
        gauss->SetParameters(amp_se, E_se, sigma);
        hist->FillRandom("g", (int)(counts * 0.15));
        
        // Add Double Escape
        gauss->SetParameters(amp_de, E_de, sigma);
        hist->FillRandom("g", (int)(counts * 0.08));
    }
    
}

void GenerateCNNTraining() {
    TFile *file = new TFile("CNN_Training_Set.root", "RECREATE");
    TTree *tree = new TTree("dataset", "Spectra with Labels for CNN");

    // --- BRANCH VARIABLES ---
    std::vector<double> b_spectrum;
    std::vector<double> b_cleanSignal;
    int b_nPeaks;
    std::vector<double> b_centroids;
    std::vector<double> b_amplitudes;
    std::vector<double> b_sigmas;

    tree->Branch("spectrum", &b_spectrum);
    tree->Branch("clean_signal", &b_cleanSignal);
    tree->Branch("n_peaks", &b_nPeaks, "n_peaks/I");
    tree->Branch("peak_centroids", &b_centroids);
    tree->Branch("peak_amplitudes", &b_amplitudes);
    tree->Branch("peak_sigmas", &b_sigmas);

    TRandom3 rnd(0);
    TH1D *hTotal = new TH1D("hTotal", "", N_BINS, 0, MAX_ENERGY);
    hTotal->SetDirectory(0);
    TH1D *hSignalOnly = new TH1D("hSignalOnly", "", N_BINS, 0, MAX_ENERGY);
    hSignalOnly->SetDirectory(0);

    TF1 *gauss = new TF1("g", "gaus", 0, MAX_ENERGY);
    TF1 *compton = new TF1("compton", "[0] * 0.5 * TMath::Erfc((x - [1]) / [2])", 0, MAX_ENERGY);
    
    TF1 *bgFunc = new TF1("bgFunc", "[0]*exp([1]*x) + [2]", 0, MAX_ENERGY);

    std::cout << "Generating " << N_EVENTS << " training samples..." << std::endl;

    for (int ev = 0; ev < N_EVENTS; ++ev) {
        hTotal->Reset();
        hSignalOnly->Reset();
        b_spectrum.clear();
        b_cleanSignal.clear();
        b_centroids.clear();
        b_amplitudes.clear();
        b_sigmas.clear();

        // Background
        double bg_amp = rnd.Uniform(200, 8000);
        double bg_decay = rnd.Uniform(-0.004, -0.001);
        double bg_const = rnd.Uniform(10, 500);
        bgFunc->SetParameters(bg_amp, bg_decay, bg_const);
        hTotal->FillRandom("bgFunc", 4000);

        // Peaks
        b_nPeaks = rnd.Integer(8) + 1;
        
        for (int i = 0; i < b_nPeaks; ++i) {
            double centroid = rnd.Uniform(MAX_ENERGY * 0.05, MAX_ENERGY * 0.95);
            double amp = rnd.Uniform(100, 10000);
            
            b_centroids.push_back(centroid);
            b_amplitudes.push_back(amp);
            b_sigmas.push_back(GetSigma(centroid));

            // Pass the persistent TF1 pointers
            AddPhysicsPeak(hTotal, hSignalOnly, centroid, amp, gauss, compton);
        }

        for (int b = 1; b <= N_BINS; ++b) {
            b_spectrum.push_back(hTotal->GetBinContent(b));
            b_cleanSignal.push_back(hSignalOnly->GetBinContent(b));
        }

        tree->Fill();

        if (ev % 1000 == 0) printf("Progress: %d / %d \n", ev, N_EVENTS);
    }

    file->Write();
    delete file; 
    
    delete bgFunc;
    delete gauss;
    delete compton;
    delete hTotal;
    delete hSignalOnly;
    
    std::cout << "Data generation complete: CNN_Training_Set.root" << std::endl;
}