from Chromatin_fixed_nucleosome import FE_fixed_nucl
import numpy as np
import argparse 
import time
import os
import h5py
# import multiprocessing

start=time.time()
parser=argparse.ArgumentParser()
parser.add_argument("-kbp", action='store', help="total length of chromatinized DNA in kbp", required=True, type=float)
parser.add_argument("-Nucl", action='store', help="Number of nucleosomes", required=True, type=int)
parser.add_argument("-nf", action='store', help="inital supercoiling density", required=True, type=int)
parser.add_argument("-ni", action='store', help="final supercoiling density", required=True, type=int)
parser.add_argument("-dn", action='store', help="number of grid points in supercoiling density", required=True, type=int)
parser.add_argument("-f", action='store', help="applied force in pN units", required=True, type=float)
parser.add_argument("-o", action='store', help="applied force in pN units", required=True, type=str)
args=parser.parse_args()

def main():
    minFE = FE_fixed_nucl(Eneg = -29.0, 
            Epos=-28.0, 
            Eopen=-30.0, 
            force=args.f, 
            L_DNA=args.kbp*340,
            wr_open=-0.7, 
            wr_neg=-1.4, 
            wr_pos=-0.4,)

    lkvals = np.arange(args.ni,args.nf,args.dn)
    sig_vals=lkvals*minFE.h/minFE.L_DNA

    fe = []
    Nop,Nneg,Npos = [],[],[]
    Lp,Lks,Lkp = [],[],[]
    wrnuc = []

    for sig in sig_vals:
        minFE.sumOverPlectAndNuclDist(sig,args.Nucl)
        fe.append(minFE.Emin)
        Nop.append(minFE.n_open)
        Nneg.append(minFE.n_neg)
        Npos.append(minFE.n_pos)
        Lp.append(minFE.l_plect)
        Lks.append(minFE.lk_stretch)
        Lkp.append(minFE.lk_plect)
        wrnuc.append(minFE.wr_nuc)

    torq = np.gradient(np.array(fe))*minFE.h/(2*np.pi*minFE.L_DNA*np.gradient(sig_vals))

    with h5py.File(os.path.join(args.o,f'Supercoiling_{args.kbp}kbp_{args.f}pN_{args.Nucl}nucl_ni{args.ni}_nf{args.nf}_dn{args.dn}.h5'),'w') as hf:
        hf.create_dataset('FE', data=np.array(fe))
        hf.create_dataset('Nop', data=np.array(Nop))
        hf.create_dataset('Npos', data=np.array(Npos))
        hf.create_dataset('Nneg', data=np.array(Nneg))
        hf.create_dataset('Torq', data=np.array(torq))
        hf.create_dataset('Lks', data=np.array(Lks))
        hf.create_dataset('Lkp', data=np.array(Lkp))
        hf.create_dataset('Wrnuc', data=np.array(wrnuc))
        hf.create_dataset('Lp', data=np.array(Lp))



if __name__=="__main__": main()