import numpy as np
import io

def WriteRAINIERgsfTable(fname, Eg, E1, M1=None, E2=None):
    '''
    Takes gamma-ray strength function f and write it in a format readable for RAINIER

    input:
    fname: outputfilename
    x: Gamma-ray energy in MeV
    E1: E1 component [in 1/MeV^3]
    M1: M1 component [in 1/MeV^3]
    E2: E2 component, default set to 0
    '''
    if M1 is None:
        M1 = np.zeros(len(Eg))

    if E2 is None:
        E2 = np.zeros(len(Eg))

    fh = open(fname, "w")

    def write_arr(arr):
        # write array to file that resembles the CERN ROOT arrays
        for i, entry in enumerate(arr):
            if i!=(len(arr)-1):
                fh.write(str(entry)+",\n")
            else:
                fh.write(str(entry)+"\n};\n")

    fh.write("#include \"TGraph.h\"\n\n")
    fh.write("double adGSF_ETable[] = {\n")
    write_arr(Eg)
    fh.write("const int nGSF_ETable = sizeof(adGSF_ETable)/sizeof(double);\n\n")

    fh.write("double adGSF_E1[] = {\n")
    write_arr(E1)
    fh.write("TGraph *grGSF_E1 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_E1);\n\n")

    fh.write("double adGSF_M1[] = {\n")
    write_arr(M1)
    fh.write("TGraph *grGSF_M1 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_M1);\n\n")

    fh.write("double adGSF_E2[] = {\n")
    write_arr(E2)
    fh.write("TGraph *grGSF_E2 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_E2);\n\n")

    print("Wrote gSF fro RAINIER to {}".format(fname))

    fh.close()


def WriteRAINIERnldTable(fname, Ex, nld, sigma, a=None):
    '''
    Takes nld and write it in a format readable for RAINIER

    input:
    fname: outputfilename
    x: Excitation energy in MeV
    nld: nld for above discretes [in 1/MeV]
    sigma: spin-cut
    a: level density parameter a
    '''
    if a is None:
        a = np.zeros(len(Ex))

    fh = open(fname, "w")

    def write_arr(arr):
        # write array to file that resembles the CERN ROOT arrays
        for i, entry in enumerate(arr):
            if i!=(len(arr)-1):
                fh.write(str(entry)+",\n")
            else:
                fh.write(str(entry)+"\n};\n")

    fh.write("#include \"TGraph.h\"\n\n")
    fh.write("double adETable[] = {\n")
    write_arr(Ex)
    fh.write("const int nETable = sizeof(adETable)/sizeof(double);\n\n")

    fh.write("double adRho[] = {")
    write_arr(nld)
    fh.write("TGraph *grRho = new TGraph(nETable,adETable,adRho);\n\n")

    fh.write("double adLDa[] = {\n")
    write_arr(a)
    fh.write("TGraph *grLDa = new TGraph(nETable,adETable,adLDa);\n\n")

    fh.write("double adJCut[] = {\n")
    write_arr(sigma)
    fh.write("TGraph *grJCut = new TGraph(nETable,adETable,adJCut);\n\n")

    print("Wrote nld fro RAINIER to {}".format(fname))

    fh.close()


def read_mama_2D(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array,
    # as well as a list containing the calibration coefficients
    # and 1-D arrays of calibrated x and y values for plotting and similar.
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    cal = {}
    with open(filename, 'r') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
        # JEM update 20180723: Changing to dict, including second-order term for generality:
        # print("calibration_line =", calibration_line, flush=True)
        cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]),
             "a0y":float(calibration_line[4]), "a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}
    # TODO: INSERT CORRECTION FROM CENTER-BIN TO LOWER EDGE CALIBRATION HERE.
    # MAKE SURE TO CHECK rebin_and_shift() WHICH MIGHT NOT LIKE NEGATIVE SHIFT COEFF.
    # (alternatively consider using center-bin throughout, but then need to correct when plotting.)
    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny-1, Ny)
    y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
    x_array = np.linspace(0, Nx-1, Nx)
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    # x_array = np.linspace(cal["a0x"], cal["a0x"]+cal["a1x"]*Nx, Nx) # BIG TODO: This is probably center-bin calibration,
    # x_array = np.linspace(a[2], a[2]+a[3]*(Ny), Ny) # and should be shifted down by half a bin?
                                                    # Update 20171024: Started changing everything to lower bin edge,
                                                    # but started to hesitate. For now I'm inclined to keep it as
                                                    # center-bin everywhere.
    return matrix, cal, y_array, x_array # Returning y (Ex) first as this is axis 0 in matrix language


def getCalibrationFromCounting(filename):
    """ Get calibration from counting.dat """
    f = open(filename)
    lines = f.readlines()
    # 19: float a0 =  -0.7800;
    # 20: float a1 =   0.1300;
    cal = np.genfromtxt(io.BytesIO(
        lines[18].encode()), dtype=object, delimiter="=")
    if cal[0] != b"float a0 ":
        raise ValueError("Could not read calibration")
    a0 = float(cal[1][:-1])
    cal = np.genfromtxt(io.BytesIO(
        lines[19].encode()), dtype=object, delimiter="=")
    if cal[0] != b"float a1 ":
        raise ValueError("Could not read calibration")
    a1 = float(cal[1][:-1])
    return a0, a1


def convertStrength(filename, a0_strength, a1_strength):
    """ Transform strength.nrm to gsf of type [x,y,yerr]"""
    data_ocl = np.loadtxt(filename)
    data_ocl_copy = data_ocl
    n_datapoints = int(len(data_ocl_copy) / 2)
    data_ocl = np.ndarray((n_datapoints, 3), dtype=float)
    for i in range(n_datapoints):
        if (data_ocl_copy[i] > 0):
            data_ocl[i, 0] = a0_strength + (a1_strength * i)
        else:
            data_ocl[i, 0] = 0
        data_ocl[i, 1] = data_ocl_copy[i]
        data_ocl[i, 2] = data_ocl_copy[i + n_datapoints]
    data_ocl = data_ocl[np.all(data_ocl, axis=1)]  # delete "0" elments
    return data_ocl


def getTransExt(myfile, a0_strength, a1_strength, Emin, Emax):
    """ Get extrapolation of the transmission coefficient type [x,y,yerr]"""
    transextfile = open(myfile)
    transextlines = transextfile.readlines()
    transextfile.close()
    Ntrans = len(transextlines)
    trans = np.zeros((Ntrans, 2))
    for i in range(Ntrans):
        trans[i, 0] = a0_strength + i * a1_strength
        trans[i, 1] = float(transextlines[i].split()[0]) / \
            (2. * np.pi * trans[i, 0]**3.)
    trans = trans[np.all((trans[:, 0] > Emin, trans[:, 0] < Emax), axis=0), :]
    return trans

def rhoCT(Ex,T,E0):
    """ Constant Temperature nld formula """
    E = Ex-E0
    return np.exp(E/T)/T
