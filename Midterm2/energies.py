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
        maximum = max(state1[0].level,state1[2].level,state2[0].level,state2[2].level)
        alpha = state1[0].level
        beta = state1[2].level
        gamma = state2[0].level
        delta = state2[2].level
        E = 0
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
    
    def create_set_of_states(self,Maxmimum_energy:int):
        #create a set of states
        #2 and 2 levels up to maxmum energy
        #2 levels cannot be the same in the same state
        #maybe set j in range (i+1,Maxmimum_energy+1)
        states = {}
        for i in range(1,Maxmimum_energy+1):
            for j in range(i+1,Maxmimum_energy+1):
                if i != j:
                    states[i,j ] = self.create_state(np.array([i,j]))
        return states
    

    def Hamiltonian(self,max_energy,g):
        states1 = self.create_set_of_states(max_energy)
        states2 = self.create_set_of_states(max_energy)

        H0 = np.eye(len(states1))
        for i,s in enumerate(states1):
            H0[i,i] = system.single_particle_energy(states1[s])

        V = np.zeros((len(states1),len(states2)))
        for i,s1 in enumerate(states1):
            for j,s2 in enumerate(states2):
                e = system.interraction_energy(states1[s1],states2[s2],1)
                V[i,j] = e
        print(H0)
        print(V)
        H = H0 - (g/2 * V)
        return H
     

        


if __name__ == "__main__":
    #try it out
    system = manybodies([])
    levels = np.array([1,2])
    state = system.create_state(levels)
    energy = system.single_particle_energy(state)

    print("Energy: ",energy)
    n = 101
    energies = np.zeros((n, 6))
    g_values = np.linspace(-1, 1, n)
    for i, g in enumerate(g_values):
        H = system.Hamiltonian(4,g)
        energies[i] = np.linalg.eigvalsh(H)
    
    for i in range(6):
        if i == 3:
            linestyle = "dotted"
            color = "magenta"
        else:
            linestyle = "-"
            color = None
        plt.plot(g_values, energies[:, i], label=f"State {i}", linestyle=linestyle, color=color, linewidth=4)
    
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title("Energy levels as a function of g")
    plt.legend()
    plt.show()