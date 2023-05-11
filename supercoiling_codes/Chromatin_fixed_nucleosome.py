import numpy as np
import itertools
from scipy.optimize import minimize_scalar

class FE_fixed_nucl:
    def __init__(self, 
        Eneg = -40.0, 
        Epos=None, 
        Eopen=None, 
        force=1.0, 
        L_DNA=1000.0,
        wr_open=-0.7, 
        wr_neg=-1.4, 
        wr_pos=-0.4,
        ):

        self.A = 50.0       # DNA bend persistence length (nm)
        self.C = 100.0      # DNA twist persistence length (nm)
        self.P = 25.0       # DNA plectoneme twist persistence length (nm)
        self.h = 3.5        # DNA helix repeat length (nm)
        self.l_nucl = 60.0       # DNA length absorbed by a nucleosome (nm)
        self.l_nucl_ext = 10.0   # free DNA length for open nucleosomes (nm)
        self.f = force * 0.25   # external force in kT/nm units
        self.L_DNA = L_DNA      # total DNA length (nm)
        self.lp_cutOff =10.0    # cut-off size for small plectoneme (nm)
        self.lp_mesh = 30.0     # mesh size for summing over plectoneme size (nm)

        # writhe of nucleosome states
        self.wo = wr_open 
        self.wp = wr_pos
        self.wn = wr_neg

        # binding energies (kT)
        self.e_neg = Eneg
        if Epos is None:
            self.e_open = self.e_neg + 15.0
        else:
            self.e_open = float(Eopen)
        if Epos is None:
            self.e_pos = self.e_neg + 20.0
        else:
            self.e_pos = float(Epos)

        #placeholder values; to be set after minimization.
        self.Emin = 'yet to run minimization'
        self.n_open = 'yet to run minization'
        self.n_neg = 'yet to run minization'
        self.n_pos = 'yet to run minization'
        self.l_plect = 'yet to run minization'
        self.lk_stretch = 'yet to run minization'
        self.lk_plect = 'yet to run minization'
        self.wr_nuc = 'yet to run minization'
        self.sigma = 'yet to run minization'

        print("=====================\nFixed nucleosome calculations:\n=====================\n")
        print("Total DNA length: {:.0f} nm\nExternal force: {:.2f} pN".format(self.L_DNA, self.f*4))
        print("DNA length absorbed per nucleosome: {:.0f} nm\n".format(self.l_nucl))
        print("--------------\nNegative nucleosomes\nBinding energy: {:.1f} kT\nWrithe: {:.2f}\nstretched DNA: 0.0 nm\n".format(self.e_neg, self.wn))            
        print("--------------\nOpen nucleosomes\nBinding energy: {:.1f} kT\nWrithe: {:.2f}\nstretched DNA: {:.1f} nm\n".format(self.e_open, self.wo, self.l_nucl_ext)) 
        print("--------------\nPositive nucleosomes\nBinding energy: {:.1f} kT\nWrithe: {:.2f}\nstretched DNA: 0.0 nm\n".format(self.e_pos, self.wp)) 
        print("=================================")

    def g_f(self): 
        R"""
        Force-extension free energy density in units kT/nm 
        """
        return self.f-np.sqrt(self.f/self.A)

    def Ecoex_fixedLp(self, s, N, lp, no, nn):
        R"""
        Free energy of the stretched-DNA plectoneme coexistence state for a given supercoiling density `s`,
        nucleosome distribution (N, no, nn), and plectoneme length (xlp*L).
        This method minimizes over linking number partition to ensure torque balance for a 
        given plectoneme and nucleosome distribution.

        returns: (minimum energy, stretched phase linking number, plectoneme phase linking number, writhe of nucleosome)
        
        *note* the input `s` is a an intensive quantity but outputs are extensive (which one may normalize if needed)
        """

        def E_stretch_plect_nuc(xx):
            R"""
            Energy of coexistence between plectoneme and stretched phase, 
            which includes the DNA tiwst energy within nucleosomes.
            *note* DNA twist modulus is the same in the stretched and nucleosome bound DNA
            Minimized over xx, the fraction of linking number in the stretched phase
            """
            lks = xx*(lk-wr_nuc)
            lkp = (1-xx)*(lk-wr_nuc)
            E_extend = 2*np.pi**2*self.C*lks**2/(ls+N*self.l_nucl) - self.g_f()*ls 
            E_plect = 2*np.pi**2*self.P*lkp**2/lp
            return  E_extend + E_plect

        def E_plect_nuc(xx):
            R"""
            Energy of coexistence between plectoneme and nucleosomes (twist energy of DNA within nucleosomes).
            Minimized over xx, the fraction of linking number contributing to twist in nucleosome-bound DNA.
            """
            lk_nuc_tw = xx*(lk-wr_nuc)
            lkp = (1-xx)*(lk-wr_nuc)
            E_nuc_tw = 2*np.pi**2*self.C*lk_nuc_tw**2/(N*self.l_nucl)
            E_plect = 2*np.pi**2*self.P*lkp**2/lp
            return  E_nuc_tw + E_plect
        
        def E_nuc_bind(): 
            R"""
            Nucleosome binding energy and force-extension energy gain for open states
            """
            return no*(self.e_open - self.g_f()*self.l_nucl_ext) + nn*self.e_neg + (N-no-nn)*self.e_pos 

        lk = s*self.L_DNA/self.h
        wr_nuc = no*self.wo + nn*self.wn + (N-no-nn)*self.wp
        L_free = self.L_DNA - N*self.l_nucl
        Emin_tot=1e10

        if lp <= self.lp_cutOff:
            # NO plectoneme, only stretched phase and nucleosome twisting + writhing
            Emin_tot = 2*np.pi**2*self.C*(lk-wr_nuc)**2/self.L_DNA - self.g_f()*L_free
            res = (Emin_tot + E_nuc_bind(), lk - wr_nuc, 0.0, wr_nuc)

        elif L_free - lp <= self.lp_cutOff: 
            if N==0:
                # NO stretched phase or nucleosomes, only plectoneme
                Eplect = 2*np.pi**2*self.P*(lk - wr_nuc)**2/lp
                res = (Eplect, 0.0, lk - wr_nuc, 0.0)
            else:
                # NO stretched phase, only plectoneme and nucleosome twisting + writhing
                #minimize over linking number partition
                min_val = minimize_scalar(E_plect_nuc, method='bounded', bounds=(0,1))
                res = (min_val.fun + E_nuc_bind(), 0.0, (lk-wr_nuc)*(1-min_val.x), wr_nuc) 
            

        else:    
            # Coexistence of stretched, plectoneme, and nucleosome phases
            ls = L_free - lp
            #minimize over linking number partition
            min_val = minimize_scalar(E_stretch_plect_nuc, method='bounded', bounds=(0,1))
            res = (min_val.fun + E_nuc_bind(), (lk-wr_nuc)*min_val.x, (lk-wr_nuc)*(1-min_val.x), wr_nuc) 

        return res

    def sumOverPlectAndNuclDist(self, s, N):
        R"""
        This method sums over all plectoneme lengths and nucleosome distribution to find 
        the fluctuation averaged quantities.
        The minimized values are stored as atttributes Emin, n_open, etc.
        """
        E_vals=[]
        no_vals,nn_vals=[],[]
        lp_vals=[]
        lks_vals=[]
        lkp_vals= []
        wrnuc_vals=[]
        L_free=self.L_DNA - N*self.l_nucl

        for lp,no,nn in itertools.product(np.arange(0,L_free,self.lp_mesh),range(N+1),range(N+1)):            
            if no+nn>N: continue
            Emin, lksmin, lkpmin, wrnuc = self.Ecoex_fixedLp(s, N, lp, no, nn)
            E_vals.append(Emin)
            no_vals.append(no)
            nn_vals.append(nn)
            lp_vals.append(lp)
            lks_vals.append(lksmin)
            lkp_vals.append(lkpmin)
            wrnuc_vals.append(wrnuc)

        E_vals = np.array(E_vals)
        Z = np.exp(-(E_vals-E_vals.min()))

        self.sigma = s
        self.Emin = np.sum(Z*E_vals)/Z.sum()
        self.n_open = np.sum(Z*np.array(no_vals))/Z.sum()
        self.n_neg = np.sum(Z*np.array(nn_vals))/Z.sum()
        self.n_pos = np.sum(Z*(N-np.array(nn_vals)-np.array(no_vals)))/Z.sum()
        self.l_plect = np.sum(Z*np.array(lp_vals))/Z.sum()
        self.lk_stretch = np.sum(Z*np.array(lks_vals))/Z.sum()
        self.lk_plect = np.sum(Z*np.array(lkp_vals))/Z.sum()
        self.wr_nuc = np.sum(Z*np.array(wrnuc_vals))/Z.sum()
