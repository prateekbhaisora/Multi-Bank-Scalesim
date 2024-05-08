import time
import numpy as np
from tqdm import tqdm

from memory.read_buffer import read_buffer as rdbuf
from memory.read_buffer_estimate_bw import ReadBufferEstimateBw as rdbuf_est
from memory.read_port import read_port as rdport
from memory.write_buffer import write_buffer as wrbuf
from memory.write_port import write_port as wrport


class double_buffered_scratchpad:
    '''
    The constructor method initializes an instance of the double_buffered_scratchpad class. 
    It initializes various attributes such as buffers for ifmap, filter, and ofmap, ports for reading and writing, and other metrics-related variables.
    '''
    def __init__(self):
        
        self.num_of_DRAM_banks = 2      # SRAM memory object will work on two DRAM banks

        self.ifmap_buf = rdbuf()
        self.filter_buf = rdbuf()
        self.ofmap_buf = wrbuf()

        self.ifmap_port = rdport()
        self.filter_port = rdport()
        self.ofmap_port = wrport()

        self.verbose = True

        # Trace matrices for SRAM
        self.ifmap_trace_matrix = np.zeros((1,1), dtype=int)    
        self.filter_trace_matrix = np.zeros((1,1), dtype=int)
        self.ofmap_trace_matrix = np.zeros((1,1), dtype=int)

        # Metrics to gather for generating run reports
        self.total_cycles = 0
        self.compute_cycles = 0
        self.stall_cycles = 0

        self.avg_ifmap_dram_bw = 0
        self.avg_filter_dram_bw = 0
        self.avg_ofmap_dram_bw = 0

        self.ifmap_sram_start_cycle = 0
        self.ifmap_sram_stop_cycle = 0
        self.filter_sram_start_cycle = 0
        self.filter_sram_stop_cycle = 0
        self.ofmap_sram_start_cycle = 0
        self.ofmap_sram_stop_cycle = 0

        self.ifmap_dram_start_cycle = 0
        self.ifmap_dram_stop_cycle = 0
        self.ifmap_dram_reads = 0
        self.filter_dram_start_cycle = 0
        self.filter_dram_stop_cycle = 0
        self.filter_dram_reads = 0
        self.ofmap_dram_start_cycle = 0
        self.ofmap_dram_stop_cycle = 0
        self.ofmap_dram_writes = 0

        self.estimate_bandwidth_mode = False,
        self.traces_valid = False
        self.params_valid_flag = True

    '''
    This method allows setting parameters for the scratchpad memory system.
    It includes parameters like verbosity, bandwidth estimation mode, buffer sizes, active buffer fractions, and bandwidths for different components.
    These parameters configure the behavior and characteristics of the scratchpad memory.
    '''
    def set_params(self,
                   verbose=True,
                   estimate_bandwidth_mode=False,
                   word_size=1,
                   ifmap_buf_size_bytes=2, filter_buf_size_bytes=2, ofmap_buf_size_bytes=2,
                   rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
                   ifmap_backing_buf_bw=1, filter_backing_buf_bw=1, ofmap_backing_buf_bw=1):

        self.estimate_bandwidth_mode = estimate_bandwidth_mode

        # IFMAP and FILTER BUffers changed to estimated if estimate_bandwidth_mode is used.
        if self.estimate_bandwidth_mode:   
            self.ifmap_buf = rdbuf_est()
            self.filter_buf = rdbuf_est()

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_default_bw=ifmap_backing_buf_bw)

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_default_bw=filter_backing_buf_bw)
        else:
            self.ifmap_buf = rdbuf()
            self.filter_buf = rdbuf()

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_bw=ifmap_backing_buf_bw)

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_bw=filter_backing_buf_bw)

        # Common OFMAP parameters
        self.ofmap_buf.set_params(backing_buf_obj=self.ofmap_port,
                                  total_size_bytes=ofmap_buf_size_bytes,
                                  word_size=word_size,
                                  active_buf_frac=wr_buf_active_frac,
                                  backing_buf_bw=ofmap_backing_buf_bw)

        self.verbose = verbose

        self.params_valid_flag = True

    '''
    This method sets prefetch matrices for the ifmap and filter buffers. 
    Prefetch matrices define the prefetching behavior for fetching data from DRAM to the scratchpad buffers.
    '''
    def set_read_buf_prefetch_matrices(self,
                                       ifmap_prefetch_mat=np.zeros((1,1)),
                                       filter_prefetch_mat=np.zeros((1,1))
                                       ):

        self.ifmap_buf.set_fetch_matrix(ifmap_prefetch_mat)
        self.filter_buf.set_fetch_matrix(filter_prefetch_mat)

    '''
    This method resets the states of all buffers in the scratchpad memory system. 
    It clears any existing data and sets the buffers to their initial states.
    '''
    def reset_buffer_states(self):

        self.ifmap_buf.reset()
        self.filter_buf.reset()
        self.ofmap_buf.reset()

    '''
    This method services read requests for ifmap data from DRAM. 
    It takes incoming requests and cycles, processes them, and returns the cycles needed to service the requests. 
    '''
    # The following are just shell methods for users to control each mem individually
    def service_ifmap_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):     
        out_cycles_arr_np = self.ifmap_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    '''
    Similar to the service_ifmap_reads method, this method services read requests for filter data from DRAM.
    The bank_id parameter specifies the DRAM bank from which to read filter data.
    '''
    def service_filter_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        out_cycles_arr_np = self.filter_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    '''
    This method services write requests for ofmap data to DRAM. 
    It takes incoming requests and cycles, processes them, and returns the cycles needed to service the requests.
    The bank_id parameter specifies the DRAM bank to which to write ofmap data.
    '''
    def service_ofmap_writes(self,
                             incoming_requests_arr_np,  # 2D array with the requests
                             incoming_cycles_arr):

        out_cycles_arr_np = self.ofmap_buf.service_writes(incoming_requests_arr_np, incoming_cycles_arr)

        return out_cycles_arr_np

    '''
    This method orchestrates the servicing of memory requests for ifmap, filter, and ofmap data. 
    It iterates over each line of the ofmap data, services the corresponding ifmap and filter read requests, and writes the resulting ofmap data.
    It calculates metrics such as total cycles and stall cycles during memory access.
    '''
    def service_memory_requests(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        # Throws an assertion error if params_valid_flag is false, i.e., memories have not been initialized properly
        assert self.params_valid_flag, 'Memories not initialized yet'

        # Determines the number of lines (requests) to be serviced in the OFMAP (output feature map) demand matrix.
        ofmap_lines = ofmap_demand_mat.shape[0]

        self.total_cycles = 0
        self.stall_cycles = 0
        
        # Retrieves the hit latencies for the IFMAP (input feature map) and filter buffers respectively. 
        # These latencies represent the time it takes to access data in the buffers when it's already present (a cache hit).
        ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
        filter_hit_latency = self.filter_buf.get_hit_latency()

        # Initialize lists to store the cycles at which IFMAP, filter, and OFMAP requests are serviced respectively.
        ifmap_serviced_cycles = []
        filter_serviced_cycles = []
        ofmap_serviced_cycles = []

        # Determines whether to display a progress bar based on the verbosity setting. 
        pbar_disable = not self.verbose
        # Iterates over each line of the OFMAP demand matrix, displaying a progress bar if enabled.
        for i in tqdm(range(ofmap_lines), disable=pbar_disable):

            # Creates a 2D array representing the cycle at which the current OFMAP line is being serviced. 
            # It takes into account any accumulated stall cycles.
            cycle_arr = np.zeros((1,1)) + i + self.stall_cycles

            # Retrieves the IFMAP demand for the current line.
            ifmap_demand_line = ifmap_demand_mat[i, :].reshape((1,ifmap_demand_mat.shape[1]))
            # Services IFMAP reads using the IFMAP buffer.
            ifmap_cycle_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=ifmap_demand_line,
                                                            incoming_cycles_arr=cycle_arr)
            # Updates the ifmap_serviced_cycles list with the cycle at which IFMAP requests are serviced.
            ifmap_serviced_cycles += [ifmap_cycle_out[0]]
            # Calculates any stalls incurred during IFMAP reads.
            ifmap_stalls = ifmap_cycle_out[0] - cycle_arr[0] - ifmap_hit_latency

            # Services Filter requests, similar to IFMAP requests.
            filter_demand_line = filter_demand_mat[i, :].reshape((1, filter_demand_mat.shape[1]))
            filter_cycle_out = self.filter_buf.service_reads(incoming_requests_arr_np=filter_demand_line,
                                                           incoming_cycles_arr=cycle_arr)
            filter_serviced_cycles += [filter_cycle_out[0]]
            filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency

            # Services OFMAP requests, similar to IFMAP and Filter requests.
            ofmap_demand_line = ofmap_demand_mat[i, :].reshape((1, ofmap_demand_mat.shape[1]))
            ofmap_cycle_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=ofmap_demand_line,
                                                             incoming_cycles_arr_np=cycle_arr)
            ofmap_serviced_cycles += [ofmap_cycle_out[0]]
            ofmap_stalls = ofmap_cycle_out[0] - cycle_arr[0] - 1

            # Calculates the maximum of the stalls incurred during IFMAP reads, filter reads, and OFMAP writes, and adds it to the total stall cycles.
            self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))

        # If estimating bandwidth mode is enabled, it completes prefetches for IFMAP and filter buffers.
        if self.estimate_bandwidth_mode:
            # IDE shows warning as complete_all_prefetches is not implemented in read_buffer class
            # It is harmless since, in estimate bandwidth mode, read_buffer_estimate_bw is instantiated
            self.ifmap_buf.complete_all_prefetches()
            self.filter_buf.complete_all_prefetches()

        # Empties all buffers in the OFMAP scratchpad based on the last serviced cycle of OFMAP requests.
        self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])

        # Prepare the traces:
        # 1) Converts the lists of serviced cycles for IFMAP, filter, and OFMAP into NumPy arrays.
        # 2) Concatenates these arrays with their respective demand matrices to form trace matrices.
        # 3) Updates the total cycles counter with the cycle at which the last OFMAP request was serviced.

        ifmap_services_cycles_np = np.asarray(ifmap_serviced_cycles).reshape((len(ifmap_serviced_cycles), 1))
        self.ifmap_trace_matrix = np.concatenate((ifmap_services_cycles_np, ifmap_demand_mat), axis=1)

        filter_services_cycles_np = np.asarray(filter_serviced_cycles).reshape((len(filter_serviced_cycles), 1))
        self.filter_trace_matrix = np.concatenate((filter_services_cycles_np, filter_demand_mat), axis=1)

        ofmap_services_cycles_np = np.asarray(ofmap_serviced_cycles).reshape((len(ofmap_serviced_cycles), 1))
        self.ofmap_trace_matrix = np.concatenate((ofmap_services_cycles_np, ofmap_demand_mat), axis=1)
        
        self.total_cycles = int(ofmap_serviced_cycles[-1][0])

        # END of serving demands from memory
        self.traces_valid = True

    '''
    This method returns the total number of cycles taken for computation, including memory access cycles.
    '''
    def get_total_compute_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.total_cycles

    '''
    This method returns the total number of stall cycles incurred during memory access.
    '''
    def get_stall_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.stall_cycles

    '''
    This method returns the start and stop cycles for IFMAP data in the scratchpad SRAM.
    '''
    def get_ifmap_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_start_cycle = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_stop_cycle  = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle

    '''
    This method returns the start and stop cycles for FILTER data in the scratchpad SRAM.
    '''
    def get_filter_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):
            if done:
                break
            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_start_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_stop_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        return self.filter_sram_start_cycle, self.filter_sram_stop_cycle

    '''
    This method returns the start and stop cycles for OFMAP data in the scratchpad SRAM.
    '''
    def get_ofmap_sram_start_stop_cycles(self):
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_start_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_stop_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle

    '''
    This method returns details about ifmap data access from DRAM, such as start and stop cycles and the number of accesses.
    The bank_id parameter specifies the DRAM bank.
    '''
    def get_ifmap_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.ifmap_dram_reads = self.ifmap_buf.get_num_accesses()
        self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle \
            = self.ifmap_buf.get_external_access_start_stop_cycles()

        return self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle, self.ifmap_dram_reads

    '''
    Similar to the above method, this method returns details about filter data access from DRAM.
    '''
    def get_filter_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.filter_dram_reads = self.filter_buf.get_num_accesses()
        self.filter_dram_start_cycle, self.filter_dram_stop_cycle \
            = self.filter_buf.get_external_access_start_stop_cycles()

        return self.filter_dram_start_cycle, self.filter_dram_stop_cycle, self.filter_dram_reads

    '''
    Similar to the above methods, this method returns details about ofmap data access to DRAM.
    '''
    def get_ofmap_dram_details(self):
        assert self.traces_valid, 'Traces not generated yet'

        self.ofmap_dram_writes = self.ofmap_buf.get_num_accesses()
        self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle \
            = self.ofmap_buf.get_external_access_start_stop_cycles()

        return self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle, self.ofmap_dram_writes

    '''
    This method returns the trace matrix for ifmap data in the scratchpad SRAM.
    '''
    def get_ifmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix

    '''
    Similar to the above method, this method returns the trace matrix for filter data in the scratchpad SRAM.
    '''
    def get_filter_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.filter_trace_matrix

    '''
    Similar to the above methods, this method returns the trace matrix for ofmap data in the scratchpad SRAM.
    '''
    def get_ofmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ofmap_trace_matrix

    def get_sram_trace_matrices(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix, self.filter_trace_matrix, self.ofmap_trace_matrix

    #
    def get_ifmap_dram_trace_matrix(self):
        return self.ifmap_buf.get_trace_matrix()

    #
    def get_filter_dram_trace_matrix(self):
        return self.filter_buf.get_trace_matrix()

    #
    def get_ofmap_dram_trace_matrix(self):
        return self.ofmap_buf.get_trace_matrix()

    #
    def get_dram_trace_matrices(self):
        dram_ifmap_trace = self.ifmap_buf.get_trace_matrix()
        dram_filter_trace = self.filter_buf.get_trace_matrix()
        dram_ofmap_trace = self.ofmap_buf.get_trace_matrix()

        return dram_ifmap_trace, dram_filter_trace, dram_ofmap_trace

    '''
    These methods prints the trace matrix for ifmap, filter and ofmap data in the scratchpad SRAM to a file.
    '''
    def print_ifmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ifmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_filter_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.filter_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ofmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_trace_matrix, fmt='%i', delimiter=",")

    '''
    These methods prints the trace matrix for ifmap, filter and ofmap data in the DRAM to a file.
    '''
    def print_ifmap_dram_1_trace(self, filename):
        self.ifmap_buf.print_trace_1(filename)

    #
    def print_ifmap_dram_2_trace(self, filename):
        self.ifmap_buf.print_trace_2(filename)

    #
    def print_filter_dram_1_trace(self, filename):
        self.filter_buf.print_trace_1(filename)

    #
    def print_filter_dram_2_trace(self, filename):
        self.filter_buf.print_trace_2(filename)

    #
    def print_ofmap_dram_trace(self, filename):
        self.ofmap_buf.print_trace(filename)