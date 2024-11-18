import numpy as np
import matplotlib.pyplot as plt

class Electron:
    def __init__(self,level:int,spin:float):
        self.level = level
        self.spin = spin
        self.particle = {'level':self.level,'spin':self.spin}

class manybodies:
    def __init__(self,particles):
        self.particles = particles
        self.manybody = {'particles':self.particles}
        try:
            self.orbitals = self.fill_orbitals()
            self.states =   self.fill_states()
        except:
            pass
        # print("State: ",self.states)
    
    def add_particle(self,particle):
        if particle not in self.particles:
            self.particles.append(particle)
            self.manybody = {'particles':self.particles}
            self.orbitals = self.fill_orbitals()
        else:
            print('Particle already exists in the system')

    def fill_orbitals(self):
        #fill the orbitals with electrons
        orbitals = {}
        for i in range(len(self.particles)):
            if self.particles[i]['level'] in orbitals:
                orbitals[self.particles[i]['level']].append(self.particles[i]['spin'])
            else:
                orbitals[self.particles[i]['level']] = [self.particles[i]['spin']]
        return orbitals

    
    def single_particle_energy(self,state):
        if len(state) != 4:
            raise ValueError('State must have 4 particles')
        energy = 0
        E_level1 = state[0].level
        E_level2 = state[2].level
        return 2*(E_level1+E_level2 -2)

    def interraction_energy(self,state1, state2,g):
        #check for orthogonality
        if len(state1) != 4 or len(state2) != 4:
            raise ValueError('State must have 4 particles')
        #find max energy level
       

        alpha = state1[0].level
        beta = state1[2].level
        gamma = state2[0].level
        delta = state2[2].level
        E = 0
        maximum = max(alpha,beta,gamma,delta)

        for p in range(1,maximum+1):
            for q in range(1,maximum+1):
                
                if beta == delta and alpha == p and q == gamma:
                    
                    E += 1
                    # return +1
                if beta == gamma and alpha == p and delta == q:
                    
                    E += 1
                    # return -1
                if beta == p and alpha == delta and gamma == q:
                    
                    E += 1
                    # return -1
                if beta == p and alpha == gamma and delta == q:
                    
                    E += 1
                    # return +1
                
        return E*g

    
    def pair_creation_operator(self,level):
        p1,p2 = Electron(level,1/2),Electron(level,-1/2)
        return p1,p2
    
    def create_state(self,levels:np.array):
        #create a state with 4 particles
        p = []
        for i in range(len(levels)):
            p1,p2 = self.pair_creation_operator(levels[i])
            p.append(p1)
            p.append(p2)
        if len(p)%4 == 0:
            return p
        else:
            print('Error in creating the state. Need to have 4 particles per state')
        #every state has 4 particle
        #create the state with 4 particle
        state = {}
        for i in range(0,len(levels),4):
            p1,p2,p3,p4 = p[i],p[i+1],p[i+2],p[i+3]
            state[i] = [p1,p2,p3,p4]
        return state
    
    def create_set_of_states(self,MaxEP1:int,MaxEp2:int):
        #create a set of states
        #2 and 2 levels up to maxmum energy
        #2 levels cannot be the same in the same state
        #maybe set j in range (i+1,Maxmimum_energy+1)
        states = {}
        for i in range(1,MaxEP1+1):
            for j in range(i+1,MaxEp2+1):
                if i != j:
                    states[i,j ] = self.create_state(np.array([i,j]))
        return states
    

    def Hamiltonian(self,MaxEp1:int,MaxEp2:int,g:np.ndarray):
        states1 = self.create_set_of_states(MaxEp1,MaxEp2)
        states2 = self.create_set_of_states(MaxEp1,MaxEp2)

        H0 = np.eye(len(states1))
        for i,s in enumerate(states1):
            H0[i,i] = system.single_particle_energy(states1[s])

        V = np.zeros((len(states1),len(states2)))
        for i,s1 in enumerate(states1):
            for j,s2 in enumerate(states2):
                e = system.interraction_energy(states1[s1],states2[s2],1)
                V[i,j] = e
        self.H0 = H0
        self.V = V
        H = self.H0 - (g/2 * self.V)
        return H
    
    def crondelta(self,a,b):
        if a == b:
            return 1
        return 0
    
    def HF(self,dims,g):
        hmat = np.zeros((dims,dims))
        alpha = np.arange(1,dims,1)
        hmat[0,0] = 2 - g
        for a in alpha:
            hmat[a,a] = ( (2 - g) + sum((p) * self.crondelta(a,p) for p in [0,1])
                            + sum((p) * self.crondelta(a,p) for p in [2,3]) 
                                + 0.5 * g * sum(1 * self.crondelta(a,p) for p in [0,1]) )
     

        return hmat
    
    def MBPT3rd(self, g):
        return 2-g -7/24*g**2 -1/12*g**3
    def MBPT2rd(self, g):
        return 2-g -7/24*g**2 
    def MBPT4rd(self, g):
        return 2-g -7/24*g**2 -1/12*g**3 -0.149*g**4

    




if __name__ == "__main__":
    #try it out
    system = manybodies([])
    levels = np.array([1,2])
    state = system.create_state(levels)
    energy = system.single_particle_energy(state)

    n = 101
    energies = np.zeros((n, 5))
    g_values = np.linspace(-1, 1, n)
    for i, g in enumerate(g_values):
        H = system.Hamiltonian(2,4,g)
        energies[i] = np.linalg.eigvalsh(H)
    
    energies2 = np.zeros((n, 6))
    for i, g in enumerate(g_values):
        H = system.Hamiltonian(3,4,g)
        energies2[i] = np.linalg.eigvalsh(H)
    

    deltaE0 = energies2[:, 0] - energies[:, 0]
    plt.plot(g_values, deltaE0, label="Ground state", linestyle="-", color="blue", linewidth=2)
    plt.xlabel("g")
    plt.ylabel("Δ E")
    plt.title("Ground state energy difference as a function of g")
    plt.legend()
    plt.show()


    # for i in range(5
    #                ):
    #     if i == 3:
    #         linestyle = "dotted"
    #         color = "magenta"
    #     else:
    #         linestyle = "-"
    #         color = None
    #     plt.plot(g_values, energies[:, i], label=f"State {i}", linestyle=linestyle, color=color, linewidth=4)
    
    # plt.xlabel("g")
    # plt.ylabel("Energy")
    # plt.title("Energy levels as a function of g")
    # plt.legend()
    # plt.show()

    
    plt.plot(g_values, energies[:,0], label="Ground state", linestyle="-", color="blue", linewidth=2)
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title("Ground state energy as a function of g")
    plt.legend()
    plt.show()

    exact = energies[:, 0]
    approx = energies2[:, 0]
    hfE = np.zeros((n,4))
    #hfEnergies = np.zeros((n,4))
    for i, g in enumerate(g_values):
        mat = system.HF(4,g)
   
        hfE[i] = np.linalg.eigvalsh(mat)

    hf = hfE[:,0]

    plt.plot(g_values, exact, label="FCI", linestyle="-", color="blue", linewidth=2)
    plt.plot(g_values, approx, label="FCI Approx", linestyle="dashed", color="red", linewidth=2)
    plt.plot(g_values, hf, label="HF", linestyle="dashdot", color="green", linewidth=2)
    plt.xlabel("g")
    plt.grid()
    plt.legend()
    plt.ylabel("Energy")
    plt.title("Ground state energy as a function of g")
    plt.show()


    RS3= np.zeros(n)
    RS2 = np.zeros(n)
    RS4 = np.zeros(n)
    for i, g in enumerate(g_values):
        RS3[i] = system.MBPT3rd(g)
        RS2[i] = system.MBPT2rd(g)
        RS4[i] = system.MBPT4rd(g)



    plt.plot(g_values, exact, label="FCI", linestyle="-", color="blue", linewidth=2)
    plt.plot(g_values, RS3, "x-",label="RS3", color="purple", linewidth=2)
    plt.plot(g_values, hf, label="HF", linestyle="dashdot", color="green", linewidth=2)

    # plt.plot(g_values, RS2, "-.",label="RS2", color="red", linewidth=2)
    # plt.plot(g_values, RS4, "--",label="RS4", color="green", linewidth=2)


    # Add labels, title, legend, and grid
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title("Ground State Energy as a Function of g")
    plt.legend()
    plt.grid()
    plt.show()

    dRS3 = exact-RS3
    dRS2 = exact-RS2
    dRS4 = exact-RS4
    dHF = exact-hf
    plt.plot(g_values, dRS3, label="RS3", linestyle="solid", color="blue", linewidth=2)
    plt.plot(g_values, dRS2, label="RS2", linestyle="dashed", color="red", linewidth=2)
    plt.plot(g_values, dRS4, "-.",label="RS4", color="green", linewidth=2)
    plt.plot(g_values, dHF, "-x",label="HF", color="purple", linewidth=2)
    plt.xlabel("g")
    plt.ylabel("Δ E")
    plt.title("Ground state energy difference as a function of g")

    plt.legend()
    plt.show()