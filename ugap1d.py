"""
    description:
    1D reaction-diffusion-heat dynamics for UV curing of urethane grafted acrylate polymer
    
    author: bhowell@berkeley.edu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

class UGAP1D:
    def __init__(self):
        # FORMULATION - WT. PERCENT
        self.percent_PI = 0.0333                                        #|   wt.%  | weight percent of photo initiator
        self.percent_PEA = 0.15                                         #|   wt.%  | weight percent of PEA
        self.percent_HDDA = 0.0168                                      #|   wt.%  | weight percent of HDDA
        self.percent_8025D = 0.4084                                     #|   wt.%  | weight percent of 8025D
        self.percent_8025E = 0.0408                                     #|   wt.%  | weight percent of 8025E
        self.percent_E4396 = 0.3507                                     #|   wt.%  | weight percent of HDDA
        self.percent_M = self.percent_PEA + self.percent_HDDA           #|   wt.%  | weight percent of monomer

        # PHYSICAL PROPERTIES
        # densities and molecular weights
        self.rho_PEA = 1.02e3                                           #| kg/m^3  | density of PEA (estimated)
        self.rho_HDDA = 1.02e3                                          #| kg/m^3  | density of HDDA (estimated)
        self.rho_E4396 = 1.10e3                                         #| kg/m^3  | density of EBECRYL 4396 
        self.rho_M = 0.899 * self.rho_PEA + 0.101 * self.rho_HDDA       #| kg/m^3  | weighted average density of monomer
        self.rho_P = 0.03 * self.rho_HDDA + \
                     0.29 * self.rho_PEA + \
                     0.68 * self.rho_E4396                              #| kg/m^3  | weighted average density of polymer
        # self.rho_P = self.rho_E4396                                   #| kg/m^3  | weighted average density of polymer
        self.rho_UGAP = 1840                                            #| kg/m^3  | estimated density of UGAP

        self.mw_PEA = 0.19221                                           #|  kg/mol | molecular weight of PEA
        self.mw_HDDA = 0.226                                            #|  kg/mol | molecular weight of HDDA
        self.mw_M = 0.899 * self.mw_PEA + 0.101 * self.mw_HDDA          #|  kg/mol | weighted average molecular weight of monomer
        self.mw_PI = 0.4185                                             #|  kg/mol | molecular weight of photo initiator
        # mw_M = mw_PEA

        self.basis_wt = 0.5                                             #|   kg    | arbtirary starting ink weight
        self.basis_vol = self.basis_wt / self.rho_UGAP                  #|   m^3   | arbitrary starting ink volume
        self.mol_PI = self.basis_wt * self.percent_PI / self.mw_PI      #|   mol   | required PI for basis weight
        self.mol_M = self.basis_wt * self.percent_M / self.mw_M         #|   mol   | required monomer for basis weight

        # diffusion properties
        self.Dm0 = 1.08e-6                                              #|  m^2/s  | diffusion constant pre-exponential, monomer (taki lit.)
        self.Am = 0.66                                                  #| unitless| diffusion constant parameter, monomer (shanna lit.)

        # BOWMAN REACTION PROPERTIES
        self.Rg = 8.3145                                                #| J/mol K | universal gas constant
        self.alpha_P = 0.000075                                         #|   1/K   | coefficent of thermal expansion, polymerization (taki + bowman lit.)
        self.alpha_M = 0.0005                                           #|   1/K   | coefficent of thermal expansion, monomer (taki + bowman lit.)
        self.theta_gP = 236.75                                          #|    K    | glass transition temperature, polymer UGAP (measured TgA)
        self.theta_gM = 313.6                                           #|    K    | glass transition temperature, monomer (Taki lit.)
        self.k_P0 = 1.145e2                                             #|m^3/mol s| true kinetic constant, polymerization (taki lit.)
        self.E_P = 10.23e3                                              #|  J/mol  | activation energy, polymerization (lit.)
        self.A_Dp = 0.05                                                #| unitless| diffusion parameter, polymerization (lit.)
        self.f_cp = 5.17e-2                                             #| unitless| critical free volume, polymerization (lit.)
        self.k_T0 = 1.337e3                                             #|m^3/mol s| true kinetic constant, termination (taki lit.)
        self.E_T = 2.94e3                                               #|  J/mol  | activation energy, termination (bowman lit.)
        self.A_Dt = 1.2                                                 #| unitless| activation energy, termination (taki lit.)
        self.f_ct = 5.81e-2                                             #| unitless| critical free volume, termination (taki lit.)
        self.R_rd = 11                                                  #|  1/mol  | reaction diffusion parameter (taki lit.)

        self.k_i = 4.8e-5                                               #|  s^-1   | primary radical rate constant

        
        # thermal properties
        self.dHp = 54.8e3                                               #|  W/mol  | heat of polymerization of acrylate monomers
        self.Cp_nacl = 880.0                                            #| J/kg/K  | heat capacity of NaCl
        self.Cp_pea = 180.3                                             #| J/mol/K | heat capacity of PEA @ 298K - https://polymerdatabase.com/polymer%20physics/Cp%20Table.html
        self.Cp_pea /= self.mw_PEA                                      #| J/kg/K  | convert units
        self.Cp_hdda = 202.9                                            #| J/mol/K | solid heat capacity of HDDA - https://webbook.nist.gov/cgi/cbook.cgi?ID=C629118&Units=SI&Mask=1F
        self.Cp_hdda /= self.mw_HDDA                                    #| J/kg/K  | convert units
        
        # # SHANNA PARAMS
        # self.shanna_c_PI0 = 150                                         #| mol/m^3 | initial PI concentration
        # self.shanna_c_M0 = 8.25e3                                       #| mol/m^3 | initial monomer concentration 
        # self.shanna_mw_M = 130.14e-3                                    #| mol/kg  | mw of monomer
        # self.shanna_mw_PI = 256.301e-3                                  #| mol/kg  | mw of PI
        self.Cp_shanna = 1700                                           #| J/kg/K  | shanna's heat capacity
        self.K_thermal_shanna = 0.2                                     #| W/m/K   | shanna's thermal conductivity

        self.eps = 9.66e-1                                              #|m^3/mol m| initiator absorbtivity
        self.phi = 0.6                                                  #| unitless| quantum yield inititation

        # solution arrays
        self.time = []
        self.sol_intensity_profile = []
        self.sol_c_PI = []

        self.sol_c_PIdot = []
        self.sol_c_Mdot = []
        self.sol_c_M = []
        self.sol_kt = []
        self.sol_kp = []
        self.sol_test = []
        self.sol_consume = []
        self.sol_diffuse = []
        self.sol_theta = []

        # numerical method parameters: backward euler
        self.tol = 6e-6
        self.thresh = 50

    def _compute_rate_constants(self, c_M: np.ndarray, theta: np.ndarray) -> tuple:
        """
            description:
                model obtained from:
                    Development of a comprehensive free radical photopolymerization model incorporating heat and mass transfer e$ects in thick films
                    
            inputs:
                c_M - concentration of monomer (mol / m^3)
                theta - temperature (K)
            outputs:
                k_p - polymerization rate constant 
                k_t - termination rate constant
                f - free volume
        """
        # compute the total fractional free volume
        vT = c_M * self.mw_M / self.rho_M + (self.c_M0 - c_M) * self.mw_M / self.rho_P                                  # bowman eqn 20
        phi_M = c_M * self.mw_M / self.rho_M / vT                                                                       # bowman eqn 22
        phi_P = (self.c_M0 - c_M) * self.mw_M / self.rho_P / vT                                                         # bowman eqn 23
        f = 0.025 + self.alpha_M * phi_M * (theta - self.theta_gM) + self.alpha_P * phi_P * (theta - self.theta_gP)     # bowman eqn 24

        # compute temperature dependent rate constants
        # BOWMAN 1  eqn 17
        const1 = (1 / f - 1 / self.f_cp)
        denom_p = (1 + np.exp(self.A_Dp * const1))
        k_P = self.k_P0 * np.exp(-self.E_P / self.Rg / theta) / denom_p

        # BOWMAN 1  eqn 18
        k_tr = self.R_rd * k_P * c_M
        const2 = -self.A_Dt * (1 / f - 1 / self.f_ct)
        denom_t = (k_tr / (self.k_T0 * np.exp(-self.E_T / self.Rg / theta)) + np.exp(const2))
        k_T = self.k_T0 * np.exp(-self.E_T / self.Rg / theta) / (1 + 1 / denom_t)

        return k_P, k_T, f
    
    def _PI_absorption(self, c_PI):
        """
                computes energy absorption of PHOTOINITIATOR

                inputs: 
                c_PI - conctration gradient along z
                I0 - UV intensity at the surface
                zspace - [0(surface), ..., L(max depth)]

            returns
                energy_absorption [(surface), ..., L(max depth)]
        """
        
        return self.eps * self.I0 * c_PI * np.exp(-self.eps * c_PI * self.zspace)
            
    def _compute_photoinitiator_rate(self, c_PI, method=1):
        """
            see Taki et el eqn (1) & (2)

            method 2 -> forward euler
            method 1 -> backward euler
        """
        rhs = lambda c_PI: -(self.phi * self.eps * self.I0 * c_PI * np.exp(-self.eps * c_PI * self.zspace)) / 2

        if method == 0:
            return c_PI + self.dt * rhs(c_PI)

        elif method == 1:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_PI)
            while error > self.tol: 
                sol_1 = c_PI + self.dt * rhs(sol_0)
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                if count > self.thresh:
                    raise Exception("---photoinitiator solution did not converge : error {} ---".format(error))
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            return sol_1

        elif method == 2:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_PI)
            while error > self.tol: 
                trap = (1 - self.psi) * rhs(c_PI) + self.psi * rhs(sol_0)
                sol_1 = c_PI + self.dt * rhs(sol_0)
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                if count > self.thresh:
                    raise Exception("---photoinitiator solution did not converge : error {} ---".format(error))
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            return sol_1

        else:
            raise Exception("---choose correct numerical scheme for _compute_photoinitiator_rate---")
        
    def _compute_primary_radical_rate(self, c_PIdot, c_PI, c_M, method=1):
        rhs = lambda c_PIdot, c_PI, c_M: self.phi * self.eps * self.I0 * c_PI * np.exp(-self.eps * c_PI * self.zspace) - self.k_i * c_PIdot * c_M
        
        if method == 0:
            return c_PIdot + self.dt * rhs(c_PIdot, c_PI, c_M)

        elif method == 1:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_PIdot)
            while error > self.tol: 
                sol_1 = c_PIdot + self.dt * rhs(sol_0, c_PI, c_M)
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                if count > self.thresh:
                    raise Exception("---primary_radical solution did not converge : error {} ---".format(error))
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            return sol_1
        
        elif method == 2:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_PIdot)
            while error > self.tol: 
                trap = (1 - self.psi) * rhs(c_PIdot, c_PI, c_M) + self.psi * rhs(sol_0, c_PI, c_M)
                sol_1 = c_PIdot + self.dt * trap
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                if count > self.thresh:
                    raise Exception("---primary_radical solution did not converge : error {} ---".format(error))
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            return sol_1

        else: 
            raise Exception("---choose correct numerical scheme for _compute_primary_radical_rate---")

    def _compute_active_chain_rate(self, c_Mdot, c_M, c_PIdot, method=1):
        #  + self.Dm0 * np.exp(-self.Am / self.f) * self.A1 @ c_Mdot / self.dz ** 2
        rhs = lambda c_Mdot, c_M, c_PIdot: self.k_i * c_PIdot * c_M - self.k_T * c_Mdot ** 2 

        if method == 0:
            return c_Mdot + self.dt * rhs(c_Mdot, c_M, c_PIdot)

        elif method == 1:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_Mdot)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---active_chain solution did not converge : error {} ---".format(error))
                sol_1 = c_Mdot + self.dt * rhs(sol_0, c_M, c_PIdot)
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                sol_0 = copy.deepcopy(sol_1)
                count += 1
            return sol_1
        
        elif method == 2:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_Mdot)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---active_chain solution did not converge : error {} ---".format(error))
                trap = (1 - self.psi) * rhs(c_Mdot, c_M, c_PIdot) + self.psi * rhs(sol_0, c_M, c_PIdot)
                sol_1 = c_Mdot + self.dt * trap
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                sol_0 = copy.deepcopy(sol_1)
                count += 1
            return sol_1

        else: 
            raise Exception("---choose correct numerical scheme for _compute_active_chain_rate---")

    def _compute_acrylate_monomer_rate(self, c_M, c_Mdot, c_PIdot, method=1):
        def rhs(c_M, c_Mdot, c_PIdot):
            Dm = self.Dm0 * np.exp(-self.Am / self.f)
            diffuse = Dm / self.dz ** 2 * self.A1 @ c_M 
            consume = self.k_P * c_Mdot * c_M + self.k_i * c_PIdot * c_M
            return diffuse - consume, diffuse, consume
        
        if method == 0:
            forward_euler, diff, cons = rhs(c_M, c_Mdot, c_PIdot)
            return (c_M + self.dt * forward_euler), diff, cons

        elif method == 1: 
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_M)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---acrylate monomer solution did not converge : error {} ---".format(error))
                output, diff, cons = rhs(sol_0, c_Mdot, c_PIdot)
                sol_1 = c_M + self.dt * output
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            # # enforce neumann bc's
            # sol_0[-1] = sol_0[-2]
            # sol_0[0] = sol_0[1]
            return sol_0, diff, cons 
        
        elif method == 2:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(c_M)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---acrylate monomer solution did not converge : error {} ---".format(error))
                forward, _, _ = rhs(c_M, c_Mdot, c_PIdot)
                backward, _, _, = rhs(sol_0, c_Mdot, c_PIdot)
                trap = (1 - self.psi) * forward + self.psi * backward
                sol_1 = c_M + self.dt * trap
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                count += 1
                sol_0 = copy.deepcopy(sol_1)
            # # enforce neumann bc's
            # sol_0[-1] = sol_0[-2]
            # sol_0[0] = sol_0[1]
            _, diff, cons = rhs(sol_0, c_Mdot, c_PIdot)
            return sol_0, diff, cons 

        else: 
            raise Exception("---choose correct numerical scheme for _compute_acrylate_monomer_rate---")
        
    def _compute_temperature_rate(self, theta, c_M, c_Mdot, c_PI, method=1):
        """
            dθ/dt = (∇ (K ∇θ) + R_p * dHp + I_abs) / (rho_ugap * Cp)
        """
        rhs = lambda theta, c_M, c_Mdot, c_PI: self.K_thermal_shanna / self.dz ** 2 * self.A1 @ theta + \
                                                self.k_P * c_M * c_Mdot * self.dHp + \
                                                self.eps * self.I0 * c_PI * np.exp(-self.eps * c_PI * self.zspace)
        if method == 0:
            return theta + self.dt * rhs(theta, c_M, c_Mdot, c_PI) / self.rho_UGAP / self.Cp_shanna
            
        elif method == 1: 
            error = 100
            count = 0
            sol_0 = copy.deepcopy(theta)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---temperature solution did not converge : error {} ---".format(error))
                sol_1 = theta + self.dt * rhs(sol_0, c_M, c_Mdot, c_PI) / self.rho_UGAP / self.Cp_shanna
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                sol_0 = copy.deepcopy(sol_1)
                count += 1

            # enforce neumann bc's
            # sol_0[-1] = sol_0[-2]
            # sol_0[0] = sol_0[1]
            return sol_0

        elif method == 2:
            error = 100
            count = 0
            sol_0 = copy.deepcopy(theta)
            while error > self.tol: 
                if count > self.thresh:
                    raise Exception("---temperature solution did not converge : error {} ---".format(error))
                
                trap = (1 - self.psi) * rhs(theta, c_M, c_Mdot, c_PI) + self.psi * rhs(sol_0, c_M, c_Mdot, c_PI)
                sol_1 = theta + self.dt * trap / self.rho_UGAP / self.Cp_shanna
                error = np.linalg.norm(sol_0 - sol_1, ord=2)
                sol_0 = copy.deepcopy(sol_1)
                count += 1

            # enforce neumann bc's
            # sol_0[-1] = sol_0[-2]
            # sol_0[0] = sol_0[1]
            return sol_0
        else: 
            raise Exception("---choose correct numerical scheme for _compute_temperature_rate---")

    def simulate(self, dt, T, nodes, L, I0, method, psi=None, tol=1e-6, temp0=303.15):
        if psi is not None:
            assert method == 2, "assert: psi is the trapezoidal scheme hyperparameter"
            assert (psi >= 0) and (psi <= 1), "assert: psi must be >= 0 and <= 1"
            self.psi = psi
        if method == 2 and psi is None:
            self.psi = 0.5 

        start_time = time.time()
        # discretize time domain
        self.dt = dt
        self.I0 = I0
        self.t_step = np.arange(0, T + self.dt, self.dt)
        
        # discretize spacial domain
        self.zspace = np.linspace(0, L, nodes)
        self.dz = self.zspace[1]

        # INITIAL CONDITIONS
        self.c_PI0 = self.mol_PI / self.basis_vol                       #| mol/m^3 | initial concentration of PI
        self.c_M0 = self.mol_M / self.basis_vol                         #| mol/m^3 | initial concentration of monomer

        c_PI_space = np.ones(nodes) * self.c_PI0
        c_PIdot_space = np.zeros(nodes)
        c_Mdot_space = np.zeros(nodes)
        c_M_space = np.ones(nodes) * self.c_M0
        theta_space = np.ones(nodes) * temp0

        # A matrix
        self.A1 = np.zeros((nodes, nodes))
        for i in range(nodes):
            if i == 0:
                self.A1[i, i] = 0
            elif i == nodes - 1:
                self.A1[i, i] = 0
            else:
                self.A1[i, i] = -2
                self.A1[i, i - 1] = 1
                self.A1[i, i + 1] = 1

        for i, t in enumerate(self.t_step): 
            if i % len(self.t_step) - 1 / 100 == 0:
                print('theta_space: ', theta_space[0])
            # compute energy intensity and reaction rate constants 303.15
            self.k_P, self.k_T, self.f = self._compute_rate_constants(c_M_space, theta_space)
            
            # step 1: compute photoinitiator rate
            c_PI_space_ = self._compute_photoinitiator_rate(c_PI_space, method=method)
            
            # step 2: compute primary radical rate
            c_PIdot_space_ = self._compute_primary_radical_rate(c_PIdot_space, c_PI_space, c_M_space, method=method)

            # step 3: compute active chain rate:
            c_Mdot_space_ = self._compute_active_chain_rate(c_Mdot_space, c_M_space, c_PIdot_space, method=method)

            # step 4: compute acrylate monomer rate:
            c_M_space_, diffuse, consume = self._compute_acrylate_monomer_rate(c_M_space, c_Mdot_space, c_PIdot_space, method=method)

            # step 5: compute temperature rate:
            theta_space_ = self._compute_temperature_rate(theta_space, c_M_space, c_Mdot_space, c_PI_space, method=method)

            if i % 1000 == 0:
                self.time.append(t)
                self.sol_c_PI.append(c_PI_space_)
                self.sol_c_PIdot.append(c_PIdot_space_)
                self.sol_c_Mdot.append(c_Mdot_space_)
                self.sol_c_M.append(c_M_space_)
                self.sol_theta.append(theta_space_)
                self.sol_kt.append(self.k_T)
                self.sol_kp.append(self.k_P)
                self.sol_consume.append(consume)
                self.sol_diffuse.append(diffuse)
                # self.sol_test.append(test)
            
            c_PI_space = c_PI_space_
            c_PIdot_space = c_PIdot_space_
            c_Mdot_space = c_Mdot_space_
            c_M_space = c_M_space_
            theta_space = theta_space_

        print('\n--- simulation complete: {}s ---'.format(time.time() - start_time))
        
    def visualize(self):
        node_top = 2
        node_bottom = -3

        self.time = np.asarray(self.time)
        # sol_intensity_profile = np.asarray(sol_intensity_profile)
        sol_c_PI = np.asarray(self.sol_c_PI)
        sol_c_PIdot = np.asarray(self.sol_c_PIdot)
        sol_c_Mdot = np.asarray(self.sol_c_Mdot)
        sol_c_M = np.asarray(self.sol_c_M)
        sol_kt = np.asarray(self.sol_kt)
        sol_kp = np.asarray(self.sol_kp)
        sol_diffuse = np.asarray(self.sol_diffuse)
        sol_consume = np.asarray(self.sol_consume)
        sol_theta = np.asarray(self.sol_theta)
        # sol_test = np.asarray(sol_test)
        # print("np.amin(sol_c_M): ", np.amin(sol_c_M))
        
        print('\n-------------- plotting -------------\n')
        fs_ = 25
        ssize = 100
        # fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
        # # axs[0, 0].scatter(self.time, sol_c_PIdot[:, 1], color='r', s=ssize, label='top layer')
        # # axs[0, 0].scatter(self.time, sol_c_PIdot[:, -2], color='b', s=ssize, label='bottom layer')
        # # # axs[0, 0].set_title(r'concentration of PI$\cdot$ ($mol/m^3$)', fontsize=fs_)
        # # axs[0, 0].set_ylabel(r'conc. $PI\cdot$ ($mol/m^3$)', fontsize=fs_-5)
        # # axs[0, 0].set_xlabel('time ($s$)', fontsize=fs_-5)
        # # axs[0, 0].set_ylim([0, 300])


        # axs[0, 0].scatter(self.time, sol_c_M[:, 0], color='r', s=ssize, label='top layer')
        # axs[0, 0].scatter(self.time, sol_c_M[:, -1], color='b', s=ssize, label='bottom layer')
        # # axs[0, 0].set_title(r'concentration of M$\cdot$ ($mol/m^3$)', fontsize=fs_)
        # axs[0, 0].set_ylabel(r'conc. $M$ ($mol/m^3$)', fontsize=fs_-5)
        # axs[0, 0].set_xlabel('time ($s$)', fontsize=fs_-5)
        # # axs[0, 0].set_ylim([0, 0.25])
        
        # axs[1, 0].scatter(self.time, sol_c_Mdot[:, 0], color='r', s=ssize, label='top layer')
        # axs[1, 0].scatter(self.time, sol_c_Mdot[:, -1], color='b', s=ssize, label='bottom layer')
        # # axs[1, 0].set_title(r'concentration of M$\cdot$ ($mol/m^3$)', fontsize=fs_)
        # axs[1, 0].set_ylabel(r'conc. $M\cdot$ ($mol/m^3$)', fontsize=fs_-5)
        # axs[1, 0].set_xlabel('time ($s$)', fontsize=fs_-5)
        # axs[1, 0].set_ylim([0, 0.25])

        # axs[0, 1].scatter(self.time, sol_diffuse[:, node_top] * 1e6, color='r', s=ssize, label='top layer')
        # axs[0, 1].scatter(self.time, sol_diffuse[:, node_bottom] * 1e6, color='b', s=ssize, label='bottom layer')
        # # axs[0, 1].set_title(r'conc of PI$\cdot$ ($mol/m^3$)', fontsize=fs_)
        # axs[0, 1].set_ylabel(r'diffusion ($\mu mol/s^2$)', fontsize=fs_-5)
        # axs[0, 1].set_xlabel('time ($s$)', fontsize=fs_-5)
        # axs[0, 1].set_ylim([-0.00025, 0.00025])

        # axs[1, 1].scatter(self.time, sol_theta[:, 0], color='r', s=ssize, label='top layer')
        # axs[1, 1].scatter(self.time, sol_theta[:, -1], color='b', s=ssize, label='bottom layer')
        # # axs[1, 1].set_title(r'concentration of M$\cdot$ ($mol/m^3$)', fontsize=fs_)
        # axs[1, 1].set_ylabel(r'temp ($K$)', fontsize=fs_-5)
        # axs[1, 1].set_xlabel('time ($s$)', fontsize=fs_-5)
        # axs[1, 1].legend(loc='lower right')
        # axs[1, 1].set_ylim([300, 345])
        # fig.tight_layout()

        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_c_PI[:, 1], color='r', s=150, label='top layer')
        plt.scatter(self.time, sol_c_PI[:, -2], color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'concentration of PI ($mol/m^3$)', fontsize=fs_)
        plt.ylabel(r'concentration ($mol/m^3$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # plot time evolution free radical concentration
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_c_PIdot[:, 1], color='r', s=150, label='top layer')
        plt.scatter(self.time, sol_c_PIdot[:, -2], color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'concentration of PI$\cdot$ ($mol/m^3$)', fontsize=fs_)
        plt.ylabel(r'concentration ($mol/m^3$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # plot time evolution of monomer dot concentration at the top/bottom surfaces:
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_c_Mdot[:, 0], color='r', s=150, label='top layer')
        plt.scatter(self.time, sol_c_Mdot[:, -1], color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'concentration of M$\cdot$ ($mol/m^3$)', fontsize=fs_)
        plt.ylabel(r'concentration ($mol/m^3$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # plot time evolution of M concentration at the top/bottom surfaces:
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_c_M[:, 1], color='r', s=150, label='top layer')
        plt.scatter(self.time, sol_c_M[:, -2], color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'concentration of M ($mol/m^3$) ', fontsize=fs_)
        plt.ylabel(r'concentration monomer ($mol/m^3$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # diffusion of M over time
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_diffuse[:, node_top] * 1e6, color='r', s=150, label='top layer')
        plt.scatter(self.time, sol_diffuse[:, node_bottom] * 1e6, color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'diffusion of monomer over time', fontsize=fs_)
        plt.ylabel(r'diffusion ($\mu mol/s^2$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # consumption of M over time
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time[:], sol_consume[:, 1], color='r', s=150, label='top layer')
        plt.scatter(self.time[:], sol_consume[:, -2], color='b', s=150, label='bottom layer')
        plt.legend(fontsize=fs_-5)
        plt.title(r'rate of consumption of monomer', fontsize=fs_)
        plt.ylabel(r'rate of consumption ($mol/s$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        # # plt.figure()
        # # plt.scatter(time, sol_test[:, 0], color='r', label='$k_T$')
        # # # plt.scatter(time, sol_test, color='r', label='$k_T$')
        # # plt.legend()
        # # plt.title(r'sol_test over time')
        # # plt.ylabel(r'sol_test')
        # # plt.xlabel('time ($s$)')
        # # plt.show()

        # plot time evolution of rate constant:
        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_kp[:, 1], color='b', s=150, label='$k_P$')
        plt.legend(fontsize=fs_-5)
        plt.title(r'$k_P$ over time', fontsize=fs_)
        plt.ylabel(r'rate constant', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.time, sol_kt[:, 1], color='r', s=150, label='$k_T$')
        plt.legend(fontsize=fs_-5)
        plt.title(r'$k_T$ over time', fontsize=fs_)
        plt.ylabel(r'rate constant', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        plt.figure(figsize=(10, 7.5))
        for i in range(len(sol_theta[:, 0])):
            if i % 10 == 0: 
                plt.scatter(self.zspace, sol_theta[i, :], s=150, label='$\theta_{}$'.format(i))
        plt.legend(fontsize=fs_-5)
        plt.title(r'$\theta$ over time', fontsize=fs_)
        plt.ylabel(r'$\theta$ ($K$)', fontsize=fs_-5)
        plt.xlabel('time ($s$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        plt.figure(figsize=(10, 7.5))
        plt.scatter(self.zspace, sol_c_M[5], color='r', s=150, label='$M$')
        plt.legend(fontsize=fs_-5)
        plt.title(r'M in space', fontsize=fs_)
        plt.ylabel(r'M', fontsize=fs_-5)
        plt.xlabel(r'z ($\mu m$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        plt.figure(figsize=(10, 7.5))
        
        for i in range(len(self.sol_diffuse)):
            if i % 4 == 0:
                plt.scatter(self.zspace[node_top:node_bottom] / 1000, sol_diffuse[i][node_top:node_bottom] * 1e6, s=150, label='diffuse {}'.format(i))
        plt.legend(fontsize=fs_-5)
        plt.title(r'diffuse in space', fontsize=fs_)
        plt.ylabel(r'diffuse', fontsize=fs_-5)
        plt.xlabel(r'z ($\mu m$)', fontsize=fs_-5)
        plt.xticks(fontsize=fs_-5)
        plt.yticks(fontsize=fs_-5)
        plt.show()

        print('done')

        # print('\n-----------------------------------------\n')

#%%
if __name__ == "__main__":
    dt = 1e-3
    T = 30.0
    nodes = 51
    L = 0.84e-3
    I0 = 10.
    tol = 1e-6
    temp0 = [303.15]

    for i in range(len(temp0)):
        print('backward euler: ')
        test1 = UGAP1D()
        test1.simulate(dt=dt, T=T, nodes=nodes, L=L, I0=I0, method=1, tol=tol, temp0=temp0[i])
        test1.visualize()

    # print('\n forward euler: ')
    # test0 = UGAP1D()
    # test0.simulate(dt, T, nodes, L, I0, method=0)
    # test0.visualize()

    # print('\ntrapezoidal method: ')
    # test2 = UGAP1D()
    # test2.simulate(dt, T, nodes, L, I0, 2)
    # test2.visualize()
    


# %%
