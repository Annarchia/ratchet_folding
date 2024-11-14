import os, functools, struct, json, math

from FoldingAnalysis.clstools import *
from FoldingAnalysis.exceptions import *

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms
import matplotlib.pyplot as plt


class BiasProperties:
    def __init__(self, 
                 time = None,               # Simulation time 
                 bias_potential = None, 
                 bias_penalty = None,      # Cumulativa ratchet force
                 cum_tot_force = None, 
                 bias_inst_force = None, 
                 z_coord = None,
                 s_coord = None, 
                 w_coord = None, 
                 z_min = None, 
                 s_min = None, 
                 w_min = None, 
                 closeness = None, 
                 progress = None):
        self.time = time
        self.bias_potential = bias_potential
        self.bias_penalty = bias_penalty
        self.cum_tot_force = cum_tot_force
        self.bias_inst_force = bias_inst_force
        self.z_coord = z_coord
        self.s_coord = s_coord
        self.w_coord = w_coord
        self.z_min = z_min
        self.s_min = s_min
        self.w_min = w_min
        self.closeness = closeness
        self.progress = progress

def ratchet_outParser(filename):
    """Function to parse ratchet.out files"""
    ratchet_out = np.loadtxt(filename)
    return BiasProperties(time=ratchet_out[:,0],
                          progress=ratchet_out[:,1],
                          closeness=ratchet_out[:,2],
                          bias_penalty=ratchet_out[:,4], # CumRatF 
                          cum_tot_force=ratchet_out[:,5])

class Structure:
    """single-frame structure
    
    Example:
        load the default structure and lazy-compute the distance matrix
        >>> dmap = Structure().dmap
        >>> type(dmap)
        <class 'numpy.ndarray'>
    """
    def __init__(self, filename_or_universe=None, name=None):
        self.u = filename_or_universe
        if not isinstance(filename_or_universe, mda.Universe):
            self.u = mda.Universe(self.u)
        self.name = name
    
    def __getattr__(self, name):
        try:
            return getattr(self.u, name)
        except AttributeError:
            return getattr(self.u.atoms, name)
    
    @lazy_property
    def darr(self):
        """amino acids distance array with 3 nearest neighbours ignored"""
        return darr(self.u.atoms, selection='name CA', ignore=3)
    
    @lazy_property
    def sarr(self):
        """
        (corresponds to cmap in em2cmp.py and FoldingAnalysis):
        distance array of CA atoms with 35 neighbours skip 
        and sigmoid_squared function applied
        """
        return sigmoid_squared(darr(self.u.atoms, selection='all', ignore=35))
    
    def save_sarr(self, filename):
        """save sarr to binary format for usage with GROMACS ratchet md"""
        save_sarr(self.sarr, filename, n_skip=35, cutoff=7.5)
    
    @lazy_property
    def dmap(self):
        """amino acids distance matrix"""
        return dmap(self.u.atoms, selection='name CA', ignore=0)
    
    @lazy_property
    def carr(self):
        """
        contacts array with 7.5A distance threshold and 3 nearest neighbours ignored
        """
        return carr(self.u.atoms, selection='name CA', ignore=3, cutoff=7.5)
    
    @lazy_property
    def carr_soft(self):
        return sigmoid_squared(self.darr)
    
    @lazy_property
    def cmap(self):
        """protein contact map with 7.5A distance threshold: symmetric square matrix"""
        return cmap(self.u.atoms, selection='name CA', ignore=0, cutoff=7.5)
    
    @lazy_property
    def cmap_soft(self):
        return sigmoid_squared(self.dmap)

class Frame:
    """
    helper class for Trajectory to handle the __getitem__ referencing
    """
    def __init__(self, traj, i):
        self.traj = traj
        self.i = i
    def __getattr__(self, name):
        try:
            return getattr(self.traj, name)[self.i]
        except (AttributeError, TypeError):
            try:
                return getattr(self.traj.trajectory[self.i], name)
            except (AttributeError, TypeError):
                return getattr(self.traj, name)
    def __getitem__(self, i):
        return Frame(self, i)

def _move_to_frame(method):
    @functools.wraps(method)
    def decorated(self, frame):
        self.trajectory[frame]
        return method(Frame(self, frame))
    return decorated

class _call_from_frame(MethodDecorator):
    def __getitem__(self, frame):
        frame = Frame(self._obj, frame)
        return lambda *a, **kw: self._method(frame, *a, **kw)

class TrajectoryFromStructure(type):	# metaclass
    def __new__(cls, clsname, parents, attrs):
        for parent in parents:
            # we exclude parents which were created with this metaclass:
            if parent.__class__ != TrajectoryFromStructure:
                # inheriting and re-decorating already defined lazy methods from parent:
                for attr_name in vars(parent):
                    if not attr_name.startswith('_'):	# important because we also inherit from Frame!
                        method = getattr(parent, attr_name)
                        try:
                            method = lazy_array(_move_to_frame(method._method))
                        except AttributeError:
                            method = _call_from_frame(method)
                        # vars()[attr_name] = method
                        attrs[attr_name] = method
        return super().__new__(cls, clsname, parents, attrs)

class Trajectory(Frame, Structure, metaclass=TrajectoryFromStructure):
    """multi-frame trajectory of a structure
    
    Examples:
        Create single-frame default trajectory and compute fraction of native contacts (Q):
        >>> t = Trajectory()
        >>> type(t.q)  # Q not computed yet
        <class 'clstools._LazyArray'>
        >>> t.q[0]     # now Q for frame #0 is computed and cached
        1.0
        >>> t[0].q is t.q[0]  # alternative way to reference a frame
        True
        >>> len(t.q)   # we've got only one frame here, so next step wouldn't dump our array
        1
        >>> t.q[:]     # the whole array of Q-s is computed and cached at this point
        array([1.])
        >>> type(t.q)  # t.q is ndarray now
        <class 'numpy.ndarray'>
        
        Structure methods work for each frame here too:
        >>> type(Trajectory()[0].dmap)
        <class 'numpy.ndarray'>
    """
    def __init__(self, 
              filename=None, 
              ref_filename=None,
              bias_properties=None,
              threshold=.3):            # in nm 
               
        ref_filename = ref_filename
        filename = filename or ref_filename
        self.u = mda.Universe(ref_filename, filename)
        self.r = mda.Universe(ref_filename)
        self.ref = Structure(self.r)
        self.ref_filename = ref_filename
        self.trajectory = self.u.trajectory
        self.filename = filename
        self.bias_properties = bias_properties
        self.name = ("-").join(os.path.splitext(filename)[0].split('/')[-3:])
        self.threshold = threshold

        if filename is None:
            dt = 0
        else:
            mdout = filename.split("/")[:-1]
            mdout.append("mdout.mdp")
            mdout = "/".join(mdout)
            with open(mdout, 'r') as f:
                while f.readline()[:5] != 'tinit':
                    continue
                timestep = float(f.readline().split(" ")[-1])
                while f.readline().strip() != '; Output frequency and precision for .xtc file':
                    continue
                dt = timestep * float(f.readline().split(" ")[-1])

        self.dt = dt	# in ps

    def __getattr__(self, name):
        try:
            return getattr(self.ref, name)
        except AttributeError:
            try:
                arr = np.array([getattr(frame, name) for frame in self.trajectory])
            except AttributeError:
                # raising custom Exception instead of AttributeError (see https://bugs.python.org/issue24983):
                raise Exception(f"{type(self).__name__} object has no attribute '{name}'")
            setattr(self, name, arr)
            return arr
    
    def __len__(self):
        return len(self.trajectory)
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @lazy_array(dumped=True)
    @_move_to_frame
    def rmsd(self):
        """protein root mean square distance (RMSD) between all atoms of a frame and the reference"""
        return rmsd(self.r.atoms, self.u.atoms)

    @lazy_array(dumped=True)
    def q(self, frame):
        """
        fraction of native contacts between the C-alpha atoms 
        (see `carr` for the definition of a contact)
        """
        return self.carr[frame][self.ref.carr].sum() / self.ref.carr.sum()
    
    @lazy_array(dumped=True)
    def q_soft(self, frame):
        return self.carr_soft[frame][self.ref.carr].sum() / self.ref.carr.sum()

    @lazy_property
    def folded(self):
        return np.any(self.rmsd[:] < self.threshold)
    
    @lazy_property
    def folding_frame(self):
        return np.where(self.rmsd[:] < self.threshold)[0][0]

    @lazy_property
    def folding_time(self):
        return np.where(self.rmsd[:] < self.threshold)[0][0] * self.dt

    def getFrameCount(self):
        """Count the number of frames in the trajectory"""
        return self.u.trajectory.n_frames
    
    def penaltyAtFolding(self):
        """
        Get the value of the ratchet force at the folding time
        """
        try:
            pen = self.bias_properties.bias_penalty
            timesteps = self.bias_properties.time
        except:
            raise BiasPropertiesError('Cannot access time and bias_penalty of bias_properties.')
        
        return pen[np.abs(timesteps-self.folding_time).argmin()]

    def penaltyAtFoldingNorm(self):
        """
        Get the value of the ratchet force at the folding time
        normalised to the total force
        """
        try:
            pen = self.bias_properties.bias_penalty
            timesteps = self.bias_properties.time
            tot_forces = self.bias_properties.cum_tot_force
        except:
            raise BiasPropertiesError('Cannot access time, bias_penalty and cum_tot_force of bias_properties.')
        return pen[np.abs(timesteps-self.folding_time).argmin()] / tot_forces[np.abs(timesteps-self.folding_time).argmin()]

    def penaltyAtTime(self, time):
        """
        Get the value of the ratchet force at an arbitrary time
        """
        try:
            pen = self.bias_properties.bias_penalty
            timesteps = self.bias_properties.time
        except:
            raise BiasPropertiesError('Cannot access the time and bias_penalty of bias_properties.')
        
        return pen[np.abs(timesteps-time).argmin()]

    def plotRMSD(self, directory, filename=None):
        """
        Plots the total RMSD along the trajectory, 
        marking the folding time (if it exists)
        """
        fig, ax = plt.subplots()
        ax.plot(np.array(range(len(self.rmsd[:])))*self.dt,self.rmsd[:], linewidth=0.8)
        ax.axhline(y=self.threshold,linewidth=0.75, linestyle='--', color='gray')
        if self.folded:
            ax.plot(self.folding_time,self.rmsd[self.folding_frame],'o',markersize=5)
            ax.axvline(x=self.folding_time,linewidth=1,linestyle=':',color='C1')                
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('RMSD ($\AA$)')
        fname = filename if filename is not None else (self.name + '_rmsd.png')
        fig.savefig(f"{directory}{fname}")
        plt.close(fig)
  
class TrajectoryEnsemble(Trajectory, Frame, Structure, metaclass=TrajectoryFromStructure):
    """
    Ensemble of trajecotries
    """
    def __init__(self, 
                 ref_filename = None, 
                 trajectories = None, 
                 dt = None):
        
        ref_filename = ref_filename
        self.r = mda.Universe(ref_filename)
        self.reference = Structure(self.r)
        
        self.trajectories = None
        self.dt = None
        
        if trajectories is not None:
            if isinstance(trajectories, Trajectory):
                self.trajectories = [trajectories]
            elif isinstance(trajectories, list) and \
            all([isinstance(trajectories[i], Trajectory) for i in range(len(trajectories))]):
                self.trajectories = trajectories
            else:
                raise TypeError('trajectories must be a Trajectory or a list of Trajectory')
        
        if dt != None:
            self.dt = dt
        else:
            DT = self.trajectories[0].dt
            self.dt = DT
        
        self.foldingTrajectories = None
    
    def __getattr__(self, name):
        try:
            return getattr(self.ref, name)
        except AttributeError:
            try:
                arr = np.array([getattr(frame, name) for frame in self.trajectory])
            except AttributeError:
                # raising custom Exception instead of AttributeError (see https://bugs.python.org/issue24983):
                raise Exception(f"{type(self).__name__} object has no attribute '{name}'")
            setattr(self, name, arr)
            return arr

    def addTrajectory(self, trajectory):
        if isinstance(trajectory, Trajectory):
            if self.trajectories is None:
                self.trajectories = [trajectory]
            else:
                self.trajectories.append(trajectory)
        elif isinstance(trajectory, list) and \
        all([isinstance(trajectory[i], Trajectory) for i in range(len(trajectory))]):
            if self.trajectories is None:
                self.trajectories = trajectory
            else:
                self.trajectories.extend(trajectory)
        else:
            raise TypeError('trajectory must be a Trajectory or a list of Trajectory')
            
        self.setReference(self.reference)
        self.foldingTrajectories = None
        
        if self.dt is not None:
            self.setDt(self.dt)

    def trajCount(self):
        return 0 if self.trajectories is None else len(self.trajectories)
    
    def setReference(self, reference):
        if reference is not None and not isinstance(reference, Trajectory):
            raise TypeError('reference must be an instance of Trajectory')
        
        self.reference = reference
        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.setReference(reference)
                
        self.foldingTrajectories = None
    
    @lazy_property  
    def rmsd_ensemble(self):
        """
        RMSD between all atoms of a frame and the reference
        for an ensemble of trajectories
        """
        RMSD = []
        for traj in self.trajectories:
            RMSD.append(traj.rmsd)
        return np.array(RMSD)
    
    @lazy_property
    def q_ensemble(self):
        """
        Fraction of native contacts for an ensemble of trajectories
        """
        Q = []
        for traj in self.trajectories:
            Q.append(traj.q[:])
        return np.array(Q)

    def getAvgCmapQ(self, n = 40, folded_only = True):
        """Placeholder: don't know what it does, not used in analysis"""
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')
        
        start = 0
        end = 1
        for traj in working_trajectories:
            if np.min(traj.q[:]) > start:
                start = np.min(traj.q[:])
            if np.max(traj.q[:]) < end:
                end = np.max(traj.q[:])

        steps = np.linspace(start, end, n)
        average_cmaps = np.zeros((n, len(working_trajectories[0].sarr[0])), dtype=np.float32)
        
        for traj in working_trajectories:
            cmaps = traj.sarr[:]
            j = 0
            for ts in steps:
                index = np.argmin(np.abs(traj.q[:] - ts))
                average_cmaps[j] += cmaps[index]
                j+=1
            print(f'> Done with {traj.name}')
        average_cmaps /= len(working_trajectories)

        return average_cmaps
    
    def getAvgCmapQBins(self, n = 40, folded_only = True):
        """Placeholder: don't know what it does, not used in analysis"""
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')
        
        start = 0
        end = 1
        for traj in working_trajectories:
            if np.min(traj.q[:]) > start:
                start = np.min(traj.q[:])
            if np.max(traj.q[:]) < end:
                end = np.max(traj.q[:])

        bins = np.linspace(start, end+0.001, n+1)
        #bins_count = np.zeros(n) #This gives invalid divide error
        bins_count = np.ones(n)

        average_cmaps = np.zeros((n, len(working_trajectories[0].sarr[0])), dtype=np.float32)
        
        for traj in working_trajectories:
            cmaps = traj.sarr[:]
            for j in range(n):
                indexs = np.logical_and(traj.q[:] >= bins[j], traj.q[:] < bins[j+1])
                bins_count[j] += np.sum(indexs)
                average_cmaps[j] += np.sum(cmaps[indexs], axis=0)
            print(f'> Done with {traj.name}')

        i = 0
        for div in bins_count:
            average_cmaps[i] /= div
            i+=1

        return average_cmaps
    
    def getAverageCmapTime(self, folded_only=True, use_ref_as_last=True, stride=1, path=None):
        """
        Computes the average contact map for an ensemble of trajectories
        """
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')

        n_frames = working_trajectories[0].u.trajectory.n_frames
        
        if not all([traj.u.trajectory.n_frames == n_frames for traj in working_trajectories]):
            raise FrameOutOfBounds('Folding trajetotires have a different number of frames')
        
        n_frames_keep = math.ceil(n_frames / stride)

        if use_ref_as_last and self.reference is not None:
            n_frames_keep += 1
        
        #average_cmaps = np.zeros((n_frames_keep, len(working_trajectories[0].sarr[0])), dtype=np.float32)
        # Create a memory-mapped array for the average contact maps
        average_cmaps = np.memmap(f'{path}/average_cmaps.mmap', dtype='float64', mode='w+', shape=(n_frames_keep, len(working_trajectories[0].sarr[0])))

        for traj in working_trajectories:
            if use_ref_as_last:
                traj_data = np.append(traj.sarr[::stride],[traj.ref.sarr[:]], axis=0)
            else:
                traj_data = traj.sarr[::stride]
            average_cmaps += traj_data
            print(f'> Done with {traj.name}')
            # clear the memory for traj_data
            del traj_data
        
        average_cmaps /= len(working_trajectories)
        # Make sure to flush changes to disk before we finish
        average_cmaps.flush()
    
    # def getAverageCmapTime(self, folded_only = True, use_ref_as_last = True):
    #     """
    #     Computes the average contact map for an ensemble of trajectories
    #     """
    #     if self.trajectories is None or len(self.trajectories) == 0:
    #         raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
    #     if self.reference is None:
    #         raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
    #     if folded_only:
    #         working_trajectories = self.getFoldedTrajectories()
    #     else:
    #         working_trajectories = self.trajectories
        
    #     if folded_only and len(working_trajectories) == 0:
    #         raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')

    #     start_f = 0
    #     end_f = working_trajectories[0].u.trajectory.n_frames

    #     n_frames = end_f - start_f
        
    #     if not all([traj.u.trajectory.n_frames == end_f for traj in working_trajectories]):
    #         raise FrameOutOfBounds('Folding trajetotires have a different number of frames')
        
    #     if use_ref_as_last and self.reference is not None:
    #         n_frames += 1
        
    #     average_cmaps = np.zeros((n_frames, len(working_trajectories[0].sarr[0])), dtype=np.float32)
        
    #     for traj in working_trajectories:
    #         if use_ref_as_last:
    #             average_cmaps += np.append(traj.sarr[:],[traj.ref.sarr[:]], axis=0)
    #         else:
    #              average_cmaps += traj.sarr[:]
    #         print(f'> Done with {traj.name}')
        
    #     average_cmaps /= len(working_trajectories)
    #     return average_cmaps
    
    def getAverageRMSDTime(self, folded_only=True):
        """Placeholder: not used in analysis"""
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')
        
        n_frames = working_trajectories[0].u.trajectory.n_frames
        
        if not all([traj.getFrameCount() == n_frames for traj in working_trajectories]):
            raise FrameOutOfBounds('Folding trajetotires have a different number of frames')
        
        average_RMSD = np.zeros(n_frames)
        for traj in working_trajectories:
            average_RMSD += traj.getRMSD()
            
        average_RMSD /= len(working_trajectories)
        
        return average_RMSD
        
    def getAverageRMSDQ(self, n=40, folded_only=True):
        """Placeholder: Not used in analysis"""
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add at least one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.getTrajectories()
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')
        
        old_ural = self.Q_settings.use_ref_as_last
        self.configureQ(use_ref_as_last = False, min_dist = self.Q_settings.min_dist, 
                        cutoff = self.Q_settings.cutoff, beta_c = self.Q_settings.beta_c, 
                        lambda_c = self.Q_settings.lambda_c)
        
        start = 0
        end = 1
        for traj in working_trajectories:
            if np.min(traj.getQ()) > start:
                start = np.min(traj.getQ())
            if np.max(traj.getQ()) < end:
                end = np.max(traj.getQ())

        steps = np.linspace(start, end, n)
        average_RMSD = np.zeros(n)
        
        for traj in working_trajectories:
            RMSD = traj.getRMSD()
            j = 0
            for ts in steps:
                index = np.argmin(np.abs(traj.getQ() - ts))
                average_RMSD[j] += RMSD[index]
                j+=1
        
        average_RMSD /= len(working_trajectories)
        
        self.configureQ(use_ref_as_last = old_ural, min_dist = self.Q_settings.min_dist, 
                        cutoff = self.Q_settings.cutoff, beta_c = self.Q_settings.beta_c, 
                        lambda_c = self.Q_settings.lambda_c)
        
        return average_RMSD
    
    def getAveragePenalty(self, folded_only=True):
        """
        Computes the average penalty for an ensemble of trajectories
        """
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
    
        if self.reference is None:
            raise MissingReference('Missing reference structure! Add one with setReference(reference)')
        
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only to False to compute the average anyway')
        
        try:
            pen_dim = len(working_trajectories[0].bias_properties.bias_penalty)
            
            if not all([len(traj.bias_properties.bias_penalty) == pen_dim for traj in working_trajectories]):
                raise FrameOutOfBounds('Folding trajetotires have a different length of bias properties')
        except:
            raise BiasPropertiesError('Cannot access the needed properties (time and bias_penalty) \
                                       of bias_properties.')
        
        average_penalty = np.zeros((len(working_trajectories),pen_dim))
        for i, traj in enumerate(working_trajectories):
            average_penalty[i] = traj.bias_properties.bias_penalty  
        
        return (np.mean(average_penalty,axis=0),np.std(average_penalty,axis=0))
        

    def getFoldedTrajectories(self):
        """
        Returns the trajectories that reach the folded state
        """
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add one with addTrajectory(trajectory)')
        
        if self.foldingTrajectories is None:
            self.foldingTrajectories = [traj for traj in self.trajectories if traj.folded]
            return self.foldingTrajectories
        else:
            return self.foldingTrajectories
        
    def getFoldedTrajectoriesCount(self):
        """
        Returns how many trajectories in the ensemble reached the folded state
        """
        if self.trajectories is None or len(self.trajectories) == 0:
            raise EmptyEnsemble('No trajectories in the ensemble. Add at least one with addTrajectory(trajectory)')
        
        if self.foldingTrajectories is None:
            self.foldingTrajectories = [traj for traj in self.trajectories if traj.folded]
            return len(self.foldingTrajectories)
        else:
            return len(self.foldingTrajectories)
        
    def plotRMSDs(self, directory):
        """
        Saves the RMSD plots for all trajectories in the ensemble
        """
        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.plotRMSD(directory)
                
    def plotMeanPenalty(self, directory, filename=None, folded_only=True):
        """
        Plots the mean bias penalty along the trajectories of an ensemble
        """
        mean_pen = self.getAveragePenalty(folded_only=folded_only)[0]
        std_pen = self.getAveragePenalty(folded_only=folded_only)[1]
        plt.plot(mean_pen)
        plt.fill_between(np.arange(len(mean_pen)),mean_pen - std_pen, mean_pen + std_pen,alpha=0.3)
        plt.xlabel('Step')
        plt.ylabel('Penalty')
        plt.grid(color='lightgray', linestyle='-.', linewidth=0.25, which='both')
        fname = filename if filename is not None else ('mean_penalty.png')
        plt.savefig(f"{directory}{fname}")
        plt.close()        
    
    def DRP(self):
        """
        Get the Dominant Reaction Pathway,
        i.e. the least biased trajectory, unormalised
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return self.getFoldedTrajectories()[np.argmin([traj.penaltyAtFolding() for traj in \
                                                       self.getFoldedTrajectories()])]
    def DRPNormal(self):
        """
        Get the Dominant Reaction Pathway,
        i.e. the least biased trajectory,
        normalised to the total force
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        
        return self.getFoldedTrajectories()[np.argmin([traj.penaltyAtFoldingNorm() for traj in \
                                                       self.getFoldedTrajectories()])]
    def DRPComplete(self):
        """
        Get the Dominant Reaction Pathway,
        i.e. the least biased trajectory,
        complete bias
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        
        max_ftime = np.max([traj.folding_time for traj in self.getFoldedTrajectories()])
        return self.getFoldedTrajectories()[np.argmin([traj.penaltyAtTime(max_ftime) for traj in \
                                                       self.getFoldedTrajectories()])]
    def maxFoldingTime(self):
        """
        Returns the maximum folding time
        of the trajectories in the ensembe
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return np.max([traj.folding_time for traj in self.getFoldedTrajectories()])
    
    def minFoldingTime(self):
        """
        Returns the minimum folding time
        of the trajectories in the ensembe
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return np.min([traj.folding_time for traj in self.getFoldedTrajectories()])
    
    def meanFoldingTime(self):
        """
        Returns the mean folding time
        of the trajectories in the ensembe
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return np.mean([traj.folding_time for traj in self.getFoldedTrajectories()])
    
    def medianFoldingTime(self):
        """
        Returns the median folding time
        of the trajectories in the ensembe
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return np.median([traj.folding_time for traj in self.getFoldedTrajectories()])
    
    def stdFoldingTime(self):
        """
        Returns the standard deviation of folding times
        of the trajectories in the ensembe
        """
        if not self.getFoldedTrajectoriesCount() > 0:
            return None
        return np.std([traj.folding_time for traj in self.getFoldedTrajectories()])
    
    def summary(self):
        """
        Returns the trajectory ensemble summary
        """
        timestep = 'NOT SET' if self.dt is None else f'{self.dt} ps'
        ref_traj = 'NOT SET' if self.reference is None else self.reference.name
        summary = ''
        summary += '\n--------------------------------------------------------------------------------\n'
        summary += '                         ~ TRAJECTORY ENSEMBLE SUMMARY ~                           '
        summary += '\n--------------------------------------------------------------------------------\n'
        summary += f'> Number of trajectories in the ensemble: {self.trajCount()} \n'
        summary += f'> Timestep: {timestep}\n'
        summary += f'> Reference structure: {ref_traj}\n'
        summary += '\n--------------------------------------------------------------------------------\n'
        if not self.getFoldedTrajectoriesCount() > 0:
            summary += '> No trajectory folded!'
            return summary
        summary += f'> A total of {self.getFoldedTrajectoriesCount()} trajectories folded\n'
        for traj in sorted(self.getFoldedTrajectories(), key=lambda x: x.folding_time):
            summary += f'> {traj.name} folded at {traj.folding_time}'
            try:
                summary += f''' | Bias: {traj.penaltyAtFolding()} | 
                            Complete Bias: {traj.penaltyAtTime(self.maxFoldingTime)} | 
                            Normalised Bias: {traj.penaltyAtFoldingNorm()}\n'''
            except:
                summary += '\n'
        
        summary += f'> Max folding time: {self.maxFoldingTime()}\n'
        summary += f'> Min folding time: {self.minFoldingTime()}\n'
        summary += f'> Mean folding time: {self.meanFoldingTime()}\n'
        summary += f'> Median folding time: {self.medianFoldingTime()}\n'
        summary += f'> Std folding time: {self.stdFoldingTime()}\n\n'
        try:
            summary += f'> DRP is {self.DRP().name} ({self.DRP().penaltyAtFolding()})\n'
            summary += f'> DRP (complete bias) is {self.DRPComplete().name} ({self.DRPComplete().penaltyAtTime(self.maxFoldingTime())})\n'
            summary += f'> DRP (normalised bias) is {self.DRPNormal().name} ({self.DRPNormal().penaltyAtFoldingNorm()})\n'
        except:
            summary += '> No DRP information available\n'
        
        return summary

    def Q_RMSD(self, file, folded_only=True):
        """
        Saves a file containgng 4 columns (frames, Q, RMSD, trajectory of origin)
        Used for energy profiling
        """
        if folded_only:
            working_trajectories = self.getFoldedTrajectories()
        else:
            working_trajectories = self.trajectories
        
        if folded_only and len(working_trajectories) == 0:
            raise EmptyEnsemble('No trajectory has folded. Set folded_only = False to compute anyway')
        
        with open(file, "w+") as f:
            if f.tell() == 0:
                f.write('Q,RMSD,Frame,Traj_index\n')
            for i, traj in enumerate(working_trajectories):
                q = traj.q_soft[:]
                rmsd = traj.rmsd[:]
                frames = np.arange(traj.getFrameCount())
                t_index = np.zeros(traj.getFrameCount())
                t_index[:] = i+1
                data = np.column_stack((q, rmsd, frames, t_index))
                np.savetxt(f,data,delimiter=',',fmt=['%.5f', '%.5f', '%d', '%d'])

################################################
    #Miscellaneous functions and utilities#
################################################

def rmsd(atoms0, atoms1):
    return 0.1 * rms.rmsd(atoms0.positions, atoms1.positions, weights=atoms1.masses, center=True, superposition=True)
    #return rms.rmsd(atoms0.positions, atoms1.positions, weights=atoms1.masses, center=True, superposition=True)

def rmsd_traj(ref_filename, traj_filename):
    reference = mda.Universe(ref_filename).atoms
    return np.array([rmsd(frame, reference) for frame in mda.Universe(ref_filename, traj_filename).trajectory])

def rmsd_traj_selection(traj:Trajectory, sel):
    """RMSD between selecion of atoms
    sel: selecion of atoms in MDAnalysys format"""
    native = traj.r.select_atoms(sel)
    return np.array([rmsd(native, traj[f].u.select_atoms(sel)) for f in range(len(traj.q[:]))])

def ratchet_to_total_force(run):
    """Ratio between ratchet and total force,
    provide a ratchet.out file"""
    data = np.loadtxt(run)
    ratchet_force = data[:, 4]
    total_force = data[:, 5]
    r_to_t = ratchet_force[1:]/total_force[1:]
    return r_to_t

def remove_neighbours_arr(arr, n=3):
    """
    remove n nearest neighbours entries from an array
    """
    if not n:
        return arr
    mask = np.zeros(len(arr), dtype=bool)
    i, k = 0, round((8*len(arr) + 1)**.5 + 1) // 2
    while i < len(arr):
        mask[i:i+n] = True
        k -= 1
        i += k
    return arr[~mask]

def remove_neighbours_mat_inplace(mat, n=None):
    if n is None:
        return mat
    np.fill_diagonal(mat, 0)
    for i in range(1, n+1):
        r = np.arange(mat.shape[0] - i)
        mat[r, r+i] = 0.
        mat[r+i, r] = 0.
    return mat

def sigmoid_squared(x):	# vectorized version adapted from `utilities.py`
    x = x * x
    cond0 = x <= 151.29
    cond1 = np.abs(x - 56.25) < 10e-5
    x *= ~cond1
    return cond0 * np.where(cond1, 0.6, (1 - (x/56.25)**3) / (1 - (x/56.25)**5))

def get_positions(atoms, selection):
    """
    returns the positions of the centers of mass
    """
    return atoms.center_of_mass(compound=selection) if selection == 'residues' else atoms.select_atoms(selection).positions

def darr(atoms, selection='all', ignore=3):
    """ 
    Returns the array of distances within a configuration
    with the first 'ignore' neares neignours removed
    """
    positions = get_positions(atoms, selection)
    return remove_neighbours_arr(distances.self_distance_array(positions, backend='OpenMP'), n=ignore)

def dmap(atoms, selection='all', ignore=3):
    pos = get_positions(atoms, selection)
    return remove_neighbours_mat_inplace(distances.distance_array(pos, pos, backend='OpenMP'), n=ignore)

def carr(atoms, selection='all', ignore=3, cutoff=7.5, soft=False): #selection='all'
    _darr = darr(atoms, selection=selection, ignore=ignore)
    if soft:
        return sigmoid_squared(_darr)
    return _darr <= cutoff

def cmap(atoms, selection='all', ignore=3, cutoff=7.5, soft=False):
    if soft:
        return sigmoid_squared(dmap(atoms, selection=selection, ignore=ignore))
    positions = get_positions(atoms, selection)
    return remove_neighbours_mat_inplace(distances.contact_matrix(positions, cutoff=cutoff), n=ignore)

def save_sarr(sarr, filename, n_skip=35, cutoff=7.5):	# adapted from `utilities.py`
    n = round((8*len(sarr) + 1)**.5 + 1) // 2 + n_skip	# recover the square matrix dimension
    mask = sum([[i, n_skip] for i in range(1, n - n_skip)], [630])	# no idea what this is...
    mat = np.empty((n, n))	# probably should rewrite this without intermediate matrix
    mat.T[np.triu_indices(n, n_skip+1)] = sarr
    sarr = mat[np.tril_indices(n, -n_skip-1)]
    # print(sarr[[j-1 + i*(2*(n-n_skip)-j)//2 for j in range(1, n-n_skip) for i in range(j)]])	# almost...
    with open(filename, 'wb') as file:
        file.write(struct.pack('=4sI4sd', b'CMAP', n, b'ABCC', cutoff))
        file.write(np.arange(n, dtype=np.uint32).tobytes())
        file.write(struct.pack('I', len(mask)))
        file.write(np.array(mask, dtype=np.uint16).tobytes())
        file.write(struct.pack('=IIdd', 1, 2, 1, 1/np.inner(sarr, sarr)))
        file.write((sarr*65535 + 0.5).astype(np.uint16).tobytes())

def q(atoms0, atoms1, cutoff=7.5):
    ref_carr = carr(atoms0, cutoff=cutoff)
    positions = atoms1.select_atoms('name CA')
    darr = remove_neighbours_arr(distances.self_distance_array(positions))
    return (darr <= cutoff)[ref_carr].sum() / ref_carr.sum()

def cmap2hard_q(cmap_traj, reference_cmap, cutoff=0.6):
    mask = reference_cmap > cutoff
    n_native = np.sum(mask)
    F = cmap_traj.shape[0]
    q = np.zeros(F)
    for i in np.arange(F):
        q[i] = np.sum(cmap_traj[i][mask] > cutoff) / n_native
    return q
