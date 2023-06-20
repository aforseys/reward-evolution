import numpy as np #adapted from Maneuver game map loading

class GridWorldMap:
    def __init__(self, map_path):
        
        self.load_map(map_path)
        print(f"map size: {self.size}x{self.size}")
        print(self.lines)
        self.agent_loc = []
        self.hazard_locs = []
        self.obstacle_locs = []
        self.target_locs = {}
        self.set_map_attributes()
        
    def get_propositions(self):   
        props = list(self.target_locs.keys())
        if self.hazard_locs: 
            props.append("h") #add h to propositions if hazards exist on map 
        return props
        
    def load_map(self, filename):
        with open(filename, 'r') as f:
            lines = [[x for x in line if x != '\n'] for line in f]
            lines = np.array(lines)
            self.lines = lines
            self.size = len(lines) #assume square map 
        
    def set_map_attributes(self):
        for idx, x in np.ndenumerate(self.lines):
            if x.lower() == "a": 
                self.agent_loc.append(np.array(idx))
            elif x.lower() == "h":
                self.hazard_locs.append(np.array(idx))
            elif x.lower() == "x":
                self.obstacle_locs.append(np.array(idx))
            elif x.isalpha(): #any other letter is target 
                assert x not in self.target_locs, "More than one target of the same type!" 
                self.target_locs[x.lower()]=np.array(idx)
                
        assert len(self.agent_loc)==1, "Each map must have one agent."
        self.agent_loc = self.agent_loc[0]
    
        
        
        
   
        