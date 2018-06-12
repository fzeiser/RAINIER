#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include "string.h"

#include <TStyle.h>
#include "TFile.h"
#include "TMath.h"
#include "TString.h"
#include "TF1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "th22mama.C"

// Runs Automatically from RAINIER.C
// Can also be run after terminal closed. 
// Run in the RAINIERversion/ directory:
// $ root Analyze.C++
// things get a bit complicated when saving info to a ROOT file 
// and retrieving parameters and globals from a text file, so if this is 
// confusing u can write your own software to execute immediately after the run

/////////////////////////////// Get Parameter From File ////////////////////////
double GetPar(string sFile, string sPar) {
  ifstream parFile;
  parFile.open(sFile.c_str());
  string sLine;
  while( getline(parFile, sLine) ) {
    istringstream issLine(sLine);
    string sTmpPar;
    double dPar;
    issLine >> sTmpPar >> dPar;
    if(sTmpPar == sPar) return dPar;
  } // getline
  return -1.0;
} // GetPar

/////////////////////////////// Get Array From File ////////////////////////////
template <class T>
void GetArray(string sFile, string sArray, T *aArray, int nLength) {
  ifstream parFile;
  parFile.open(sFile.c_str());
  string sLine;
  while( getline(parFile, sLine) ) {
    istringstream issLine(sLine);
    string sTmpPar;
    double dPar;
    issLine >> sTmpPar;
    if(sTmpPar == sArray) {
      for(int i=0; i<nLength; i++) {
        double dPar;
        issLine >> dPar;
        aArray[i] = dPar;
      } // index elements
    } // param match
  } // getline
} // GetArray

/////////////////////////////// Retrieve Parameters ////////////////////////////
const int nRunNum = 1; // which save file to pull from
double dECrit, dPlotSpMax, dExISpread, dExIMax;
int nReal, nExIMean, nEvent, nPopLvl, nDRTSC, nDisLvlMax;
bool bIsEvenA;
int *anPopLvl, *anDRTSC;
double *adDisSp, *anDisPar, *adExIMean, *adDisEne;
TF1 *fnECrit;
TFile *fSaveFile;
TGraphErrors *agrDRTSC;
void RetrievePars() {
  TString sParFile = TString::Format("Param%04d.dat",nRunNum);
  string sParam = sParFile.Data();
  dECrit     = GetPar(sParam, "g_dECrit");
  dExISpread = GetPar(sParam, "g_dExISpread");
  dExIMax    = GetPar(sParam, "g_dExIMax");
  dPlotSpMax = GetPar(sParam, "g_dPlotSpMax");
  nReal      = GetPar(sParam, "g_nReal");
  nEvent     = GetPar(sParam, "g_nEvent");
  bIsEvenA   = GetPar(sParam, "g_bIsEvenA");
  
  nDRTSC     = GetPar(sParam, "g_nDRTSC");
  anDRTSC    = new int[nDRTSC];
  GetArray(sParam, "g_anDRTSC", anDRTSC, nDRTSC);

  nPopLvl    = GetPar(sParam, "g_nPopLvl");
  anPopLvl   = new int[nPopLvl];
  GetArray(sParam, "g_anPopLvl", anPopLvl, nPopLvl);

  nDisLvlMax = GetPar(sParam, "g_nDisLvlMax");
  anDisPar= new double[nDisLvlMax];
  GetArray(sParam, "g_anDisPar", anDisPar, nDisLvlMax);

  adDisSp = new double[nDisLvlMax];
  GetArray(sParam, "g_adDisSp", adDisSp, nDisLvlMax);

  nExIMean   = GetPar(sParam, "g_nExIMean");
  adExIMean = new double[nExIMean];
  GetArray(sParam, "g_adExIMean", adExIMean, nExIMean);

  adDisEne = new double[nDisLvlMax];
  GetArray(sParam, "g_adDisEne", adDisEne, nDisLvlMax);

  agrDRTSC = new TGraphErrors[nReal*nExIMean];

  fnECrit = new TF1("fnECrit", Form("%f", dECrit), -dPlotSpMax, 
    dPlotSpMax);
  TString sSaveFile = TString::Format("Run%04d.root",nRunNum);
  fSaveFile = new TFile(sSaveFile,"read");
} // RetrievePars

void Analyze() { // program initialization
  RetrievePars();
}

//////////////////////////// Gamma Spectrum ////////////////////////////////////
void AnalyzeGamma(int exim0 = nExIMean - 1, int real0 = nReal -1) {
  if( (exim0<nExIMean) && (real0<nReal) ) {
    TCanvas *cGSpec= new TCanvas("cGSpec","cGSpec", 800, 650);
    cGSpec->SetLogy();

    TH1D *hGSpec = (TH1D*)fSaveFile->Get(Form("hExI%dGSpec_%d",exim0,real0));
    hGSpec->Draw("colz");
  } else { cout << "Non existent ExIMean or Realization" << endl; }
} // gammas

//////////////////////////// TSC Spectra ///////////////////////////////////////
void AnalyzeTSC(int nDisEx) {
  TCanvas *cTSC = new TCanvas("cTSC","cTSC", 800, 650);
  cTSC->SetLogy();
  for(int real=0; real<nReal; real++) {
    TH1D *hTSC = (TH1D*)fSaveFile->Get(Form("hExI%dto%dTSC_%d",0,nDisEx,real));

    hTSC->SetLineColor(real+2);
    hTSC->GetXaxis()->SetTitleSize(0.055);
    hTSC->GetXaxis()->SetTitleFont(132);
    hTSC->GetXaxis()->SetTitleOffset(0.8);
    hTSC->GetXaxis()->CenterTitle();
    hTSC->GetXaxis()->SetTitle("E_{#gamma} (MeV)");

    hTSC->GetYaxis()->SetTitleSize(0.055);
    hTSC->GetYaxis()->SetTitleFont(132);
    hTSC->GetYaxis()->SetTitleOffset(0.85);
    hTSC->GetYaxis()->CenterTitle();
    hTSC->GetYaxis()->SetTitle("Counts");
    hTSC->Draw("same");
  } // real
} // TSC

//////////////////////////// Internal Conversion Electron Analysis /////////////
// Not Implemented: need to separately read BrIcc XL components and electronic 
//   shell energies, but could be done with mods to RAINIER's BrIcc parser
//void AnalyzeIC(int exim0 = nExIMean - 1, int real0 = nReal -1) {
//  TCanvas *cICSpec= new TCanvas("cICSpec","cICSpec", 800, 650);
//  cICSpec->SetLogy();
//  ahICSpec[real0][exim0]->Draw("colz");
//} // gammas

//////////////////////////// Oslo Analysis /////////////////////////////////////
void AnalyzeOslo(int exim0 = nExIMean - 1, int real0 = nReal -1) {
  if( (exim0<nExIMean) && (real0<nReal) ) {
    // if bExSpread: Analysis for a specific excitation bin
    // if bExFullRxn: Already all bins
    if(exim0>=0) {
      TCanvas *cExEg = new TCanvas("cExEg","cExEg", 800, 650);
      cExEg->SetLogz();
      TH2D *h2ExEg = (TH2D*)fSaveFile->Get(Form("h2ExI%dEg_%d",exim0,real0));
      h2ExEg->Draw("colz");
      th22mama(h2ExEg,"ExEg.m");

      TCanvas *c1Gen= new TCanvas("c1Gen","c1Gen", 800, 650);
      c1Gen->SetLogz();
      TH2D *h21Gen = (TH2D*)fSaveFile->Get(Form("h2ExI%d1Gen_%d",exim0,real0));
      h21Gen->Draw("colz");
      th22mama(h21Gen,"1Gen.m");
    }
    // Analyse Oslo for all excitation energy bins (eg of bExSpread)
    if(exim0<0) {
      cout << "AnalyzeOslo: Add spectra off all excitation energies!" << endl;
      TH2D *h2ExEg;
      TH2D *h21Gen;
      for(int iexim = 0; iexim < nExIMean; iexim++){
          if (iexim==0) {
            TCanvas *cExEg = new TCanvas("cExEg","cExEg", 800, 650);
            cExEg->SetLogz();
            h2ExEg = (TH2D*)fSaveFile->Get(Form("h2ExI%dEg_%d",iexim,real0));
            h2ExEg->Draw("colz");

            TCanvas *c1Gen= new TCanvas("c1Gen","c1Gen", 800, 650);
            c1Gen->SetLogz();
            h21Gen = (TH2D*)fSaveFile->Get(Form("h2ExI%d1Gen_%d",iexim,real0));
            h21Gen->Draw("colz");
          }
          else{
            TH2D *h2ExEg1 = (TH2D*)fSaveFile->Get(Form("h2ExI%dEg_%d",iexim,real0));
            h2ExEg1->Draw("colz");
            h2ExEg->Add(h2ExEg1);
            h2ExEg->Draw("colz");

            TH2D *h21Gen1 = (TH2D*)fSaveFile->Get(Form("h2ExI%d1Gen_%d",iexim,real0));
            h21Gen1->Draw("colz");
            h21Gen->Add(h21Gen1);
            h21Gen->Draw("colz");
            // th22mama(h21Gen,"1Gen.m");
          }
          if(iexim==nExIMean-1){
            th22mama(h2ExEg,"ExEg.m");
            th22mama(h21Gen,"1Gen.m");
          }
        }
      }
    }
  else { cout << "Non existent ExIMean or Realization" << endl; }
} // Oslo

/////////////////////////////// All Populations ////////////////////////////////
void AnalyzePop(int exim0 = nExIMean - 1, int real0 = nReal - 1) {
  if( (exim0<nExIMean) && (real0<nReal) ) {
    TCanvas *cPop = new TCanvas("cPop","cPop", 800, 650);
    cPop->SetLogz();
    TH2D *h2PopLvl0=(TH2D*)fSaveFile->Get(Form("h2ExI%dPopLvl_%d",exim0,real0));
    h2PopLvl0->Draw("colz");
    fnECrit->Draw("Same");
  } else { cout << "Non existent ExIMean or Realization" << endl; }
} // Pop

//////////////////////////// Initial Populations ///////////////////////////////
void AnalyzePopDist() {
  TH2D *h2PopDist = (TH2D*)fSaveFile->Get("h2PopDist");

  if(h2PopDist) {
    h2PopDist->GetXaxis()->SetTitleSize(0.055);
    h2PopDist->GetXaxis()->SetTitleFont(132);
    h2PopDist->GetXaxis()->SetTitleOffset(0.8);
    h2PopDist->GetXaxis()->CenterTitle();

    h2PopDist->GetYaxis()->SetTitleSize(0.055);
    h2PopDist->GetYaxis()->SetTitleFont(132);
    h2PopDist->GetYaxis()->SetTitleOffset(0.65);
    h2PopDist->GetYaxis()->CenterTitle();

    h2PopDist->GetZaxis()->SetTitleSize(0.045);
    h2PopDist->GetZaxis()->SetTitleFont(132);
    h2PopDist->GetZaxis()->SetTitleOffset(-0.4);
    h2PopDist->GetZaxis()->CenterTitle();

    h2PopDist->GetXaxis()->SetTitle("J_{I} (#hbar)");
    h2PopDist->GetYaxis()->SetTitle("E_{x,I} (MeV)");
    h2PopDist->GetZaxis()->SetTitle("Relative Counts");

    TCanvas *cPopDist = new TCanvas("cPopDist", "cPopDist", 580, 650);
    cPopDist->SetLogz();

    h2PopDist->Draw("COLZ");
    fnECrit->Draw("SAME");
  } else { cout << "No Input Distribution, only for FullRxn mode" << endl; }
}

void AnalyzePopI(int exim0 = nExIMean - 1, int real0 = nReal -1) {
  if( (exim0<nExIMean) && (real0<nReal) ) {
    TCanvas *cPopI = new TCanvas("cPopI","cPopI", 800,650);
    cPopI->SetLogz();
    TH2D *h2PopI = (TH2D*)fSaveFile->Get(Form("h2ExI%dPopI_%d",exim0,real0) );
    h2PopI->Draw("COLZ");
    fnECrit->Draw("SAME");
  } else { cout << "Non existent ExIMean or Realization" << endl; }
} // initial pop

////////////////////////////// Spin Populations ////////////////////////////////
void AnalyzeJPop(int exim0 = nExIMean - 1, int real0 = nReal -1) {
  if( (exim0<nExIMean) && (real0<nReal) ) {
    TGraphErrors *agrJPop[nReal][nExIMean];
    TH2D *ahJPop  [nReal][nExIMean];
    TH1D *ahDisPop[nReal][nExIMean];
    for(int real=0; real<nReal; real++) {
      for(int exim=0; exim<nExIMean; exim++) {
        ahJPop[real][exim] = (TH2D*)fSaveFile->Get(
          Form("hExI%dJPop_%d",exim,real));
        ahJPop[real][exim]->Scale( 1 / double(nEvent) );
        agrJPop[real][exim] = new TGraphErrors(nPopLvl);

        ahDisPop[real][exim] = (TH1D*)fSaveFile->Get(
          Form("hExI%dDisPop_%d",exim,real));
        for(int lvl=0; lvl<nPopLvl; lvl++) {
          int nLvl = anPopLvl[lvl];
          int nLvlJ = adDisSp[nLvl];
          int nLvlPop = ahDisPop[real][exim]->GetBinContent(nLvl+1);

          agrJPop[real][exim]->SetPoint(lvl, nLvlJ + 0.5,
            nLvlPop / double(nEvent) );
          agrJPop[real][exim]->SetPointError(lvl, 0,
            sqrt(nLvlPop) / double(nEvent));
        } // lvl
      } // exim
    } // real

    TCanvas *cJPop = new TCanvas("cJPop","cJPop",800,650);
    // plot one initial J distribution
    ahJPop[real0][exim0]->SetLineColor(kBlack);
    ahJPop[real0][exim0]->SetLineWidth(2);
    ahJPop[real0][exim0]->GetXaxis()->SetTitle("J (#hbar)");
    ahJPop[real0][exim0]->GetYaxis()->SetTitle("Population");
    ahJPop[real0][exim0]->GetXaxis()->SetNdivisions(10,0,0,kFALSE);
    ahJPop[real0][exim0]->GetYaxis()->SetRangeUser(0,0.24);
    ahJPop[real0][exim0]->Draw();
    TLegend *legJ = new TLegend(0.673,0.620,0.889,0.890);
    int anColor[] = {kRed, kGreen, kBlue, kMagenta, kOrange, kBlack, kYellow};

    TH1D *hJIntrins = (TH1D*)fSaveFile->Get("hJIntrins");
    hJIntrins->SetLineColor(kOrange);
    hJIntrins->SetLineWidth(2);
    hJIntrins->Draw("same");

    TLegend *legI = new TLegend(0.673,0.6,0.889,0.5);
    legI->AddEntry(ahJPop[real0][exim0],"J_{I} Dist","L");
    legI->AddEntry(hJIntrins,"Underlying J Dist","L");
    legI->Draw("same");

    for(int real=0; real<nReal; real++) {
      agrJPop[real][exim0]->SetLineColor(anColor[real]);
      agrJPop[real][exim0]->SetLineWidth(2);
      agrJPop[real][exim0]->Draw("same LPE");
      legJ->AddEntry(agrJPop[real][exim0],
        Form("Realization %d",real),"LPE");
    }
    legJ->Draw("SAME");
  } else { cout << "Non existent ExIMean or Realization" << endl; }
} // J Pop

/////////////////////////// Primary 2+ TSC /////////////////////////////////////
void ScaleDRTSC(int exim, double dScale = 0.75) {
  for(int real=0; real<nReal; real++) {
    int nIndex = real + nReal * exim;
    for(int prim2=0; prim2<nDRTSC; prim2++) {
      double dEg, dY;
      agrDRTSC[nIndex].GetPoint(prim2, dEg, dY);
      double dEgErr = agrDRTSC[nIndex].GetErrorX(prim2);
      double dYErr  = agrDRTSC[nIndex].GetErrorY(prim2);
      agrDRTSC[nIndex].SetPoint(prim2, dEg, dY * dScale);
      agrDRTSC[nIndex].SetPointError(prim2, dEgErr, dYErr * dScale);
    } // prim2
  } // real
} // scale DRTSC

void AnalyzeDRTSC() { // Primary to select levels
  int nMaxDRTSC = 0;
  for(int real=0; real<nReal; real++) {
    for(int exim=0; exim<nExIMean; exim++) {
      int nIndex = real + nReal * exim;
      for(int prim2=0; prim2<nDRTSC; prim2++) {
        TH1D *hDRTSC = (TH1D*)fSaveFile->Get(
          Form("hExI%dDRTSC_%d_%d",exim,prim2,real));
        int nDRTSCInt = hDRTSC->Integral();
        if(nDRTSCInt > nMaxDRTSC) nMaxDRTSC = nDRTSCInt;
        double dEi = adExIMean[exim];
        double dEf = adDisEne[anDRTSC[prim2]];
        double dEg = dEi - dEf;
        double dGSF = nDRTSCInt / pow(dEg,3);
          // dont have a convienent scale; dont know tot width or
          // cross sec for abs mag; diff Ei have diff vals
          // going to normalize later anyways
        agrDRTSC[nIndex].SetPoint(prim2, dEg, dGSF);
        agrDRTSC[nIndex].SetPointError(prim2, dExISpread,
          sqrt(nDRTSCInt) / pow(dEg,3));
          // need to put error on dEg spread and density
      } // prim2
    } // exim
  } // real
    
  TCanvas *cGrDRTSC = new TCanvas("cGrDRTSC","cGrDRTSC",800,650);
  cGrDRTSC->SetLogy();
  TF1 *fE1M1E2 = new TF1("fE1M1E2", // assume a const temp
    "GetStrE1(5.0,x)/x**3 + GetStrM1(x)/x**3 + GetStrE2(x)/x**5",0,dExIMax);
    // Strength functions will only work immediately after the RAINIER.C run
  TH1D *hfEmpty = new TH1D("hfEmpty","hfEmpty",1000,0,20.0);
  hfEmpty->GetXaxis()->SetTitleSize(0.055);
  hfEmpty->GetXaxis()->SetTitleFont(132);
  hfEmpty->GetXaxis()->SetTitleOffset(0.8);
  hfEmpty->GetXaxis()->CenterTitle();

  hfEmpty->GetYaxis()->SetTitleSize(0.055);
  hfEmpty->GetYaxis()->SetTitleFont(132);     
  hfEmpty->GetYaxis()->SetTitleOffset(0.8);
  hfEmpty->GetYaxis()->CenterTitle();

  hfEmpty->GetXaxis()->SetTitle("E_{#gamma} (MeV)");
  hfEmpty->GetYaxis()->SetTitle("f(E_{#gamma}) (arb)");
  hfEmpty->GetXaxis()->SetRangeUser(0.0,8.0);
  hfEmpty->GetYaxis()->SetRangeUser(1e-9,1e-7);
  hfEmpty->Draw();

  TLegend *legPrim = new TLegend(0.377,0.606,0.642,0.877);
  legPrim->AddEntry(fE1M1E2,"Input f(E_{#gamma})","L");
  for(int real=0; real<nReal; real++) {
    for(int exim=0; exim<nExIMean; exim++) {
      int nIndex = real + nReal * exim;
      ///// Normalize /////
      double dMeanX = agrDRTSC[nIndex].GetMean(1); 
      double dMeanY = agrDRTSC[nIndex].GetMean(2);
      double dfEg = fE1M1E2->Eval(dMeanX);
      ScaleDRTSC(exim, dfEg / dMeanY); // not perfect, have to scale a bit more
      
      agrDRTSC[nIndex].SetLineColor(exim+2);
      agrDRTSC[nIndex].SetMarkerColor(exim+2);
      agrDRTSC[nIndex].SetMarkerStyle(20+exim);
      agrDRTSC[nIndex].SetMarkerSize(1.5);
      agrDRTSC[nIndex].SetLineWidth(2);
      agrDRTSC[nIndex].Draw("same LPE");
      double dExI = adExIMean[exim];
      legPrim->AddEntry(&agrDRTSC[nIndex],Form("E_{x,I} = %2.0f MeV",dExI),"P");
    } // exim
  } // real
  fE1M1E2->SetLineColor(kBlack);
  fE1M1E2->Draw("same"); // want on top
  legPrim->Draw();
} // DRTSC

/////////////////////////// Feeding Analysis ///////////////////////////////////
void AnalyzeFeed(int exim0 = nExIMean - 1, int real0 = nReal -1) {
  TCanvas *cFeed = new TCanvas("cFeed","cFeed", 800, 650);
  cFeed->SetLogz();
  TH2D *h2FeedTime = (TH2D*)fSaveFile->Get(
    Form("h2ExI%dFeedTime_%d",exim0,real0) );
  h2FeedTime->Draw("colz");
  TGraphErrors *agrLvlFeed[nReal][nDisLvlMax];
  TH1D *hLvlTimeProj[nReal][nExIMean][nDisLvlMax];

  double dFeedTimeMax = 220;
  TF1 *expo2 = new TF1("expo2", "[0]*exp(-x/[1]) + [2]*exp(-x/[3])",
    0.001, dFeedTimeMax);
  expo2->SetLineColor(kBlack);

  TH2D *ah2FeedTime[nReal][nExIMean];
  for(int real=0; real<nReal; real++) {
    for(int exim=0; exim<nExIMean; exim++) {
      ah2FeedTime[real][exim] = (TH2D*)fSaveFile->Get(
        Form("h2ExI%dFeedTime_%d",exim,real) );
    } // exim
  } // real

  TCanvas *cFit = new TCanvas("cFit","cFit", 100,100);
  for(int real=0; real<nReal; real++) {
    for(int lvl=0; lvl<nDisLvlMax; lvl++) {
      agrLvlFeed[real][lvl] = new TGraphErrors(nExIMean);

      for(int exim=0; exim<nExIMean; exim++) {
        double dExIMean = adExIMean[exim];

        hLvlTimeProj[real][exim][lvl]
          = (TH1D*)ah2FeedTime[real][exim]->ProjectionY(
            Form("hEx%dFeed%d_%d",exim,lvl,real), lvl, lvl+1);

        double dMaxCount = hLvlTimeProj[real][exim][lvl]->GetMaximum();
        expo2->SetParameter(0,dMaxCount/1.3);
        expo2->SetParameter(1,1.0);
        expo2->SetParameter(2,dMaxCount/3.0);
        expo2->SetParameter(3,20.0);
        hLvlTimeProj[real][exim][lvl]->Fit(expo2,"q","goff");
        hLvlTimeProj[real][exim][lvl]->GetYaxis()->SetTitleSize(0.055);
        hLvlTimeProj[real][exim][lvl]->GetYaxis()->SetTitleFont(132);
        hLvlTimeProj[real][exim][lvl]->GetYaxis()->SetTitleOffset(0.8);
        hLvlTimeProj[real][exim][lvl]->GetYaxis()->CenterTitle();
        hLvlTimeProj[real][exim][lvl]->GetYaxis()->SetTitle("Counts");
        double a0  = expo2->GetParameter(0); // dStep1Mag
        double da0 = expo2->GetParError(0);  // dStep1MagErr
        double a1  = expo2->GetParameter(1); // dStep1LifeT
        double da1 = expo2->GetParError(1);  // dStep1LifeTErr

        double a2  = expo2->GetParameter(2); // dStep2Mag
        double da2 = expo2->GetParError(2);  // dStep2MagErr
        double a3  = expo2->GetParameter(3); // dStep2LifeT
        double da3 = expo2->GetParError(3);  // dStep2LifeTErr

        double dFeedTimeMean = (a0*pow(a1,2) + a2*pow(a3,2))/(a0*a1 + a2*a3);
        // Mathematica is the best way to get this:
        // tmean[a0_, a1_, a2_, a3_] := (a0*a1^2 + a2*a3^2)/(a0*a1 + a2*a3); 
        // CForm[ FullSimplify[
        //  D[tmean[a0, a1, a2, a3], a0]^2*da0^2 + 
        //  D[tmean[a0, a1, a2, a3], a1]^2*da1^2 + 
        //  D[tmean[a0, a1, a2, a3], a2]^2*da2^2 + 
        //  D[tmean[a0, a1, a2, a3], a3]^2*da3^2]]
        // then Power -> pow, and tack on a sqrt
        double dFeedTimeMeanErr = sqrt(
          (pow(a1,2)*pow(a2,2)*pow(a1 - a3,2)*pow(a3,2)*pow(da0,2) +
          pow(a0,2)*pow(a0*pow(a1,2) + a2*(2*a1 - a3)*a3,2)*pow(da1,2) +
          pow(a0,2)*pow(a1,2)*pow(a1 - a3,2)*pow(a3,2)*pow(da2,2) +
          pow(a2,2)*pow(a0*a1*(a1 - 2*a3) - a2*pow(a3,2),2)*pow(da3,2))
          /pow(a0*a1 + a2*a3,4));

        // T_1/2 = log(2) * lifetime
        agrLvlFeed[real][lvl]->SetPoint(exim, dExIMean, dFeedTimeMean);
        agrLvlFeed[real][lvl]->SetPointError(
          exim,dExISpread,dFeedTimeMeanErr);
      } // initial excitation mean
    } // lvl 
  } // realization
  cFit->Close();

  TCanvas *cFeedMean = new TCanvas("cFeedMean","cFeedMean", 800,650);
  // TGraphErrors give bad plot control, empty histogram is better in ROOT
  TH1D *hPlotEmpty = new TH1D("hPlotEmpty","Level Feed Times",
    300, 0, dExIMax * 1.1);
  hPlotEmpty->GetXaxis()->SetTitleSize(0.055);
  hPlotEmpty->GetXaxis()->SetTitleFont(132);
  hPlotEmpty->GetXaxis()->SetTitleOffset(0.8);
  hPlotEmpty->GetXaxis()->CenterTitle();
  hPlotEmpty->GetXaxis()->SetRangeUser(3.0, 10.5);

  hPlotEmpty->GetYaxis()->SetTitleSize(0.055);
  hPlotEmpty->GetYaxis()->SetTitleFont(132);
  hPlotEmpty->GetYaxis()->SetTitleOffset(0.8);
  hPlotEmpty->GetYaxis()->CenterTitle();

  hPlotEmpty->GetXaxis()->SetTitle(
    "Initial Excitation Energy #bar{E}_{x,I} (MeV)");
  hPlotEmpty->GetYaxis()->SetTitle("Avg. Feeding Time #bar{t} (fs)");
  hPlotEmpty->GetYaxis()->SetRangeUser(0.0, 77.0);
  hPlotEmpty->Draw();

  TLegend *legFeed = new TLegend(0.9,0.7,0.7,0.9,"Fed Lvls");
  int anLvlCheck[] = {3,8,10}; // only measure these feed times experimentally
  const int nLvlCheck = sizeof(anLvlCheck) / sizeof(int);
  // plot last real; declared earlier

  for(int chk=0; chk<nLvlCheck; chk++) {
    int lvl = anLvlCheck[chk];
    agrLvlFeed[real0][lvl]->SetLineColor(chk+1);
    agrLvlFeed[real0][lvl]->SetMarkerStyle(21+chk);
    agrLvlFeed[real0][lvl]->SetMarkerSize(1.5);
    agrLvlFeed[real0][lvl]->SetMarkerColor(chk+1);
    agrLvlFeed[real0][lvl]->SetLineWidth(2);
    agrLvlFeed[real0][lvl]->Draw("same LPE");
    if(anDisPar[lvl] == 1) {
      legFeed->AddEntry(agrLvlFeed[real0][lvl],
        Form("%2.3f MeV %1.1f+", adDisEne[lvl], adDisSp[lvl]), "PE");
    } else {
      legFeed->AddEntry(agrLvlFeed[real0][lvl],
        Form("%2.3f MeV %1.1f-", adDisEne[lvl], adDisSp[lvl]), "PE");
    } // parity
  } // lvl
  legFeed->Draw("Same");

  TCanvas *cExFeed = new TCanvas("cExFeed","cExFeed",800,650);
  cExFeed->SetLogy();
  TLegend *legExFeed = new TLegend(0.9,0.52,0.7,0.9,"#bar{E}_{x,I} (MeV)");
  int nLvl = 3; // lvl to plot
  for(int exim=0; exim<nExIMean; exim++) {
    hLvlTimeProj[real0][exim][nLvl]->GetXaxis()->SetTitleOffset(0.8);
    hLvlTimeProj[real0][exim][nLvl]->SetLineColor(exim+2);
    hLvlTimeProj[real0][exim][nLvl]->SetLineWidth(2);
    legExFeed->AddEntry(hLvlTimeProj[real0][exim][nLvl],
      Form("%2.1f MeV", adExIMean[exim] ), "L");
    hLvlTimeProj[real0][exim][nLvl]->Draw("same");
  }
  legExFeed->Draw("same");
} // Feed
