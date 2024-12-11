#!/usr/bin/env python
# coding: utf-8

import sys
import traceback
import cplex

# =====================
# Data Handling Class
# =====================
class FCTPData():
    """
    This class reads and stores the data for the Fixed Cost Transportation Problem.
    You must provide:
    - S: Set of supply nodes, with supply amounts S_s
    - D: Set of demand nodes, with demand amounts D_d
    - f_sd: fixed costs for using route (s,d)
    - c_sd: variable costs per unit for route (s,d)
    Compute M_sd = min{S_s, D_d} for each pair (s,d).
    """

    def __init__(self, supply, demand, f, c):
        # supply: list or array [S_s]
        # demand: list or array [D_d]
        # f, c: matrices indexed by s, d
        # Example:
        # supply = [S_0, S_1, ...]
        # demand = [D_0, D_1, ...]
        # f[s][d], c[s][d]

        self.supply = supply
        self.demand = demand
        self.f = f
        self.c = c
        self.num_s = len(supply)
        self.num_d = len(demand)

        # Compute M_sd
        self.M = []
        for s in range(self.num_s):
            self.M.append([])
            for d in range(self.num_d):
                self.M[s].append(min(supply[s], demand[d]))


# =====================
# Worker LP (Subproblem)
# =====================
class WorkerLP():
    """
    The WorkerLP sets up the dual of the subproblem:

    Dual variables: alpha_s, gamma_d, omega_sd >= 0

    Constraints:
    -alpha_s + gamma_d - omega_sd <= c_sd   for all s,d

    Objective:
    max sum_s(-S_s * alpha_s) + sum_d(D_d * gamma_d) + sum_{s,d}(-M_sd * y_sd)*omega_sd
    """

    def __init__(self, data):
        self.data = data
        num_s = data.num_s
        num_d = data.num_d
        cpx = cplex.Cplex()
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        # Disable presolve for clarity, use primal simplex
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.primal)
        cpx.objective.set_sense(cpx.objective.sense.maximize)

        # Create variables alpha_s, gamma_d, omega_sd
        # Order: alpha_s (for s in S), gamma_d (for d in D), omega_sd (for s in S, d in D)
        alpha_idx = []
        for s in range(num_s):
            alpha_idx.append(cpx.variables.get_num())
            cpx.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[cplex.infinity],
                              names=[f"alpha.{s}"])

        gamma_idx = []
        for d in range(num_d):
            gamma_idx.append(cpx.variables.get_num())
            cpx.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[cplex.infinity],
                              names=[f"gamma.{d}"])

        omega_idx = []
        for s in range(num_s):
            row = []
            for d in range(num_d):
                idx = cpx.variables.get_num()
                row.append(idx)
                cpx.variables.add(obj=[0.0],
                                  lb=[0.0],
                                  ub=[cplex.infinity],
                                  names=[f"omega.{s}.{d}"])
            omega_idx.append(row)

        self.alpha = alpha_idx
        self.gamma = gamma_idx
        self.omega = omega_idx

        # Add constraints:
        # -alpha_s + gamma_d - omega_sd <= c_sd  for all s,d
        # rewrite: (-1)*alpha_s + (1)*gamma_d + (-1)*omega_sd <= c_sd
        for s in range(num_s):
            for d in range(num_d):
                thevars = [alpha_idx[s], gamma_idx[d], omega_idx[s][d]]
                thecoefs = [-1.0, 1.0, -1.0]
                rhs = data.c[s][d]  # c_sd
                cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                    senses=["L"],
                    rhs=[rhs])

        self.cpx = cpx
        self.cut_lhs = None
        self.cut_rhs = None

    def separate(self, y_sol):
        """
        Given a candidate y solution, solve the subproblem:
        Objective:
        max sum_s(-S_s alpha_s) + sum_d(D_d gamma_d) + sum_{s,d}(-M_sd * y_sd) omega_sd
        """

        cpx = self.cpx
        data = self.data
        num_s = data.num_s
        num_d = data.num_d

        # Update objective coefficients:
        # alpha_s: coefficient = -S_s
        # gamma_d: coefficient = D_d
        # omega_sd: coefficient = -M_sd * y_sd

        # First reset objective
        # We'll set linear obj for alpha, gamma, omega:
        # alpha_s index: self.alpha[s]
        # gamma_d index: self.gamma[d]
        # omega_sd index: self.omega[s][d]
        alpha_obj = [(-data.supply[s]) for s in range(num_s)]
        gamma_obj = [(data.demand[d]) for d in range(num_d)]
        omega_obj = []
        for s in range(num_s):
            for d in range(num_d):
                omega_obj.append(-data.M[s][d]*y_sol[s][d])

        indices = self.alpha + self.gamma + [self.omega[s][d] for s in range(num_s) for d in range(num_d)]
        cpx.objective.set_linear(zip(indices, alpha_obj + gamma_obj + omega_obj))

        # Solve the worker LP
        cpx.solve()

        # Check if unbounded (which might indicate a violated cut)
        status = cpx.solution.get_status()
        if status == cpx.solution.status.unbounded:
            # Extract an unbounded ray to form the Benders cut
            ray = cpx.solution.advanced.get_ray()
            # ray corresponds to a direction in dual space:
            # variables order: alpha_s, gamma_d, omega_sd

            # Construct the cut:
            # Benders cut form:
            # z >= sum_{s,d} f_sd y_sd + [the dual contribution terms]
            # Actually from subproblem we have:
            # max dual -> if unbounded, we get ray direction
            # The cut is:
            # sum_{s,d} (M_sd * (ray for omega_sd)) y_sd >= sum_s(-S_s * ray_alpha_s) + sum_d(D_d * ray_gamma_d)
            # We'll rearrange the cut as in the given master problem form.

            # Extract ray components
            ray_alpha = ray[0:num_s]
            ray_gamma = ray[num_s:num_s+num_d]
            ray_omega = ray[num_s+num_d:]

            # Compute LHS (coefs for y)
            # For the Benders cut:
            # LHS: sum_{s,d} (-M_sd * ray_omega_sd) y_sd
            # RHS: sum_s(-S_s * ray_alpha_s) + sum_d(D_d * ray_gamma_d)
            # We actually want:
            # z >= sum_{s,d} f_sd y_sd + [this expression from duals]

            # However, the given master problem form requires a linearization:
            # According to the given formula:
            # z >= sum_{s,d} f_sd y_sd + sum_s(-S_s alpha_s) + sum_d(D_d gamma_d) + sum_{s,d}(-M_sd omega_sd y_sd)
            #
            # From the ray we have 'directions' which define a cut of form:
            # sum_{s,d}(-M_sd * ray_omega_sd)*y_sd + sum_s(-S_s * ray_alpha_s) + sum_d(D_d * ray_gamma_d) <= 0
            # Rearranged:
            # z >= sum_{s,d} f_sd y_sd + [ ... ] (We will add this after)
            #
            # Here we assume we construct a cut using the format given:
            # We'll treat the dual variables as if they correspond to one of the stored points. 
            # So the cut is:
            # z >= sum_{s,d} f_sd * y_sd + ( sum_s(-S_s * ray_alpha_s) + sum_d(D_d * ray_gamma_d) + sum_{s,d}(-M_sd * ray_omega_sd)*y_sd )
            #
            # But we must also add the constraint:
            # sum_s(-S_s * ray_alpha_s) + sum_d(D_d * ray_gamma_d) + sum_{s,d}(-M_sd * ray_omega_sd)*y_sd <= 0
            #
            # So we get two constraints or we embed them into one big cut. 
            # The provided master problem form uses the sets of known dual solutions (hats). 
            # Here we mimic that and just add the resulting single cut that is violated.

            # Build cut lhs for y variables
            cut_vars_list = []
            cut_coefs_list = []
            idx = 0
            sum_rhs = 0.0
            # sum_s(-S_s * ray_alpha_s)
            for s in range(num_s):
                sum_rhs += (-data.supply[s])*ray_alpha[s]

            # sum_d(D_d * ray_gamma_d)
            for d in range(num_d):
                sum_rhs += data.demand[d]*ray_gamma[d]

            # sum_{s,d}(-M_sd * ray_omega_sd)* y_sd
            # also build LHS part for y
            for s in range(num_s):
                for d in range(num_d):
                    w_val = ray_omega[s*num_d + d]
                    if abs(-data.M[s][d]*w_val) > 1e-9:
                        cut_vars_list.append((s,d))
                        cut_coefs_list.append(-data.M[s][d]*w_val)

            # The final cut:
            # z >= sum_{s,d} f_sd y_sd + sum_s(-S_s ray_alpha_s) + ... + sum_{s,d}(-M_sd ray_omega_sd)*y_sd
            # => z - sum_{s,d} f_sd y_sd - [ ... ] >= 0
            # We'll build it as:
            # z - sum_{s,d}(f_sd + M_sd * ray_omega_sd)*y_sd >= -(sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d))
            # Actually, we must keep consistent with the given form. The given master problem form is:
            #
            # z >= sum_{s,d} f_sd y_sd + sum_s(-S_s alpha_s) + sum_d(D_d gamma_d) + sum_{s,d}(-M_sd omega_sd)*y_sd
            #
            # Replacing (alpha_s, gamma_d, omega_sd) with ray values:
            # This gives us a linear inequality in terms of y variables and z.
            #
            # Let's define:
            # cut_lhs: [z] + sum_{y_sd}( ... ) y_sd
            # We know z should appear with coefficient 1.
            #
            # LHS: z + sum_{s,d} [(-M_sd*ray_omega_sd) + f_sd] y_sd + ... 
            # Actually, from the provided master form, f_sd y_sd is always known, so we should also include them.
            #
            # Let's incorporate f_sd directly into the cut:
            #
            # Final:
            # z >= sum_{s,d} f_sd y_sd + [sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) + sum_{s,d}(-M_sd ray_omega_sd)*y_sd]
            #
            # Group terms in y_sd:
            # y_sd terms: f_sd y_sd + (-M_sd ray_omega_sd) y_sd = (f_sd - M_sd ray_omega_sd)*y_sd
            #
            # RHS = - ( sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) )
            #
            # Actually we have sign confusion. Let's carefully do:
            #
            # Start from:
            # sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) + sum_{s,d}(-M_sd ray_omega_sd)*y_sd <= 0
            #
            # Move this to:
            # sum_{s,d}(-M_sd ray_omega_sd)*y_sd <= - ( sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) )
            #
            # Add f_sd y_sd to both sides:
            # sum_{s,d}(f_sd - M_sd ray_omega_sd)*y_sd <= sum_{s,d} f_sd y_sd - ( sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) )
            #
            # Now add z:
            # z >= sum_{s,d} f_sd y_sd + sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d) + sum_{s,d}(-M_sd ray_omega_sd)*y_sd
            #
            # We just need to add a constraint in master to represent this. 
            #
            # Constructing final cut:
            # LHS: z - [sum_{s,d}(f_sd - M_sd ray_omega_sd)*y_sd]
            # RHS: sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d)
            #
            # We'll store LHS as cplex SparsePair with z and y variables, and set rhs appropriately.

            # We'll need to return:
            # cut: z + ( ... )y_sd >= [some RHS]
            # Let's compute final LHS and RHS:

            final_cut_vars = []
            final_cut_coefs = []

            # Add z with coef 1
            # We'll assume we have z as a global variable from master. We'll pass it similarly to how x was passed in ATSP.
            # The code structure implies we have access to the master variables in the callback. We'll do similarly.
            # We'll store them in callback class as well.
            # For demonstration we assume 'z' and 'y_sd' arrays known in callback.

            # Actually, since we define the cut here, we must just store coefs. The callback will get them and add cut.
            # We know callback passes y and z arrays to separate().
            # Let's store them in self variables and return True.

            self.cut_lhs = (cut_vars_list, cut_coefs_list, sum_rhs)
            # We'll store partial info and let callback assemble the final constraint in the callback.

            return True
        return False

# =====================
# Callback Class
# =====================
class FCTPCallback():
    """
    Similar to ATSPCallback, but for FCTP.
    We'll have:
    - Master variables: y_{sd}, plus a variable z.
    - In the invoke method, depending on context, we separate cuts at fractional or integer solutions.
    """

    def __init__(self, num_threads, data, z, y):
        self.num_threads = num_threads
        self.data = data
        self.z = z
        self.y = y
        self.workers = [None]*num_threads

    def separate_user_cuts(self, context, worker):
        # fractional solution
        # get current solution
        num_s = self.data.num_s
        num_d = self.data.num_d
        sol_y = []
        for s in range(num_s):
            row = context.get_relaxation_point(self.y[s])
            sol_y.append(row)

        if worker.separate(sol_y):
            # Construct and add user cut
            cut_vars_list, cut_coefs_list, sum_rhs = worker.cut_lhs
            # Build full cut:
            # z + sum_{s,d}( (f_sd - M_sd ray_omega_sd)*y_sd ) >= sum_s(-S_s ray_alpha_s)+... = sum_rhs
            #
            # Wait, we have not incorporated f_sd into worker. We must do it now:
            # Actually from the worker LP approach:
            # The worker returned us partial info. We must incorporate f_sd now.
            #
            # In the partial logic above, we ended with self.cut_lhs = (cut_vars_list, cut_coefs_list, sum_rhs)
            # cut_coefs_list corresponds to (-M_sd * ray_omega_sd)
            #
            # We must add f_sd to these coefficients since final cut: 
            # z >= sum_{s,d}(f_sd y_sd) + sum_s(-S_s alpha_s)+...+ sum_{s,d}(-M_sd * ray_omega_sd)*y_sd
            #
            # = sum_rhs + sum_{s,d}(M_sd * ray_omega_sd)*y_sd
            #
            # Actually, to simplify:
            # Let's say cut_coefs_list are the coefs for y_sd in the final cut relative to omega terms.
            # Add f_sd to each corresponding arc:
            final_vars = [self.z]
            final_coefs = [1.0]
            # Add each y_sd with coef = f_sd + that from cut_coefs_list
            idx = 0
            for (s,d) in cut_vars_list:
                new_coef = self.data.f[s][d] + cut_coefs_list[idx]
                final_vars.append(self.y[s][d])
                final_coefs.append(new_coef)
                idx += 1

            # Now the RHS is sum_rhs. Actually, from derivation:
            # z >= sum_s(-S_s alpha_s)+ ... + sum_{s,d}(f_sd y_sd) + sum_{s,d}(-M_sd omega_sd y_sd)
            # The worker computed sum_rhs = sum_s(-S_s ray_alpha_s) + sum_d(D_d ray_gamma_d).
            # This is the right-hand side offset we must maintain.
            # The final form:
            # z + sum_{s,d}((f_sd - M_sd ray_omega_sd)*y_sd) >= sum_rhs
            #
            # We already integrated f_sd into final_coefs. 
            # Wait, we must ensure consistency:
            # The original formula given for Master problem's cut:
            #  z >= sum_{s,d} f_sd y_sd + sum_s(-S_s alpha_s) + sum_d(D_d gamma_d) + sum_{s,d}(-M_sd omega_sd) y_sd
            #
            # The worker gives us ray for these duals. sum_rhs already includes the alpha_s and gamma_d terms. 
            # After adding f_sd into coefficients, the LHS matches exactly the final formula for the cut. 
            # So RHS = sum_rhs.

            context.add_user_cut(cplex.SparsePair(final_vars, final_coefs), 'G', sum_rhs,
                                 cutmanagement=context.use_cut.purge, local=False)

    def separate_lazy_constraints(self, context, worker):
        # integer solution
        # get candidate solution
        num_s = self.data.num_s
        num_d = self.data.num_d
        sol_y = []
        for s in range(num_s):
            sol_y.append(context.get_candidate_point(self.y[s]))

        if worker.separate(sol_y):
            # Construct lazy cut similarly
            cut_vars_list, cut_coefs_list, sum_rhs = worker.cut_lhs

            final_vars = [self.z]
            final_coefs = [1.0]

            idx = 0
            for (s,d) in cut_vars_list:
                new_coef = self.data.f[s][d] + cut_coefs_list[idx]
                final_vars.append(self.y[s][d])
                final_coefs.append(new_coef)
                idx += 1

            context.reject_candidate(constraints=[cplex.SparsePair(final_vars, final_coefs)],
                                     senses='G', rhs=[sum_rhs])

    def invoke(self, context):
        try:
            thread_id = context.get_int_info(cplex.callbacks.Context.info.thread_id)
            cid = context.get_id()
            if cid == cplex.callbacks.Context.id.thread_up:
                self.workers[thread_id] = WorkerLP(self.data)
            elif cid == cplex.callbacks.Context.id.thread_down:
                self.workers[thread_id] = None
            elif cid == cplex.callbacks.Context.id.relaxation:
                # user cuts
                self.separate_user_cuts(context, self.workers[thread_id])
            elif cid == cplex.callbacks.Context.id.candidate:
                # lazy constraints
                self.separate_lazy_constraints(context, self.workers[thread_id])
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise

# =====================
# Master Problem Creation
# =====================
def create_master_ilp(cpx, data, y, z):
    """
    Create the master ILP:
    Variables:
    y_{sd} in {0,1}
    z continuous variable representing upper bound on objective.

    Initial constraints: 
    z >= sum_{s,d} f_sd y_sd
    This gives a starting lower bound.

    We rely on Benders cuts added later.
    """

    num_s = data.num_s
    num_d = data.num_d

    # Add z variable
    zidx = cpx.variables.get_num()
    cpx.variables.add(obj=[0.0], lb=[-cplex.infinity], ub=[cplex.infinity], names=["z"])
    z.append(zidx)

    # Add y_{sd} binary variables
    for s in range(num_s):
        row = []
        for d in range(num_d):
            yidx = cpx.variables.get_num()
            cpx.variables.add(obj=[data.f[s][d]], lb=[0.0], ub=[1.0], types=["B"], names=[f"y.{s}.{d}"])
            row.append(yidx)
        y.append(row)

    # Add a trivial inequality:
    # z >= sum_{s,d} f_sd y_sd
    # This ensures z is at least as large as immediate fixed costs.
    thevars = [zidx]
    thecoefs = [1.0]
    rhs = 0.0
    for s in range(num_s):
        for d in range(num_d):
            thevars.append(y[s][d])
            thecoefs.append(-data.f[s][d])
    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                               senses=["G"],
                               rhs=[rhs])

# =====================
# Main Solve Function
# =====================
def fctp_benders(decompose_fractional, supply, demand, f, c):
    data = FCTPData(supply, demand, f, c)

    cpx = cplex.Cplex()
    num_threads = cpx.get_num_cores()
    cpx.parameters.threads.set(num_threads)

    y = []
    z = []
    create_master_ilp(cpx, data, y, z)
    z_var = z[0]

    # Create callback
    fctpcb = FCTPCallback(num_threads, data, z_var, y)

    contextmask = cplex.callbacks.Context.id.thread_up | cplex.callbacks.Context.id.thread_down | cplex.callbacks.Context.id.candidate
    if decompose_fractional:
        contextmask |= cplex.callbacks.Context.id.relaxation

    cpx.set_callback(fctpcb, contextmask)
    cpx.solve()

    solution = cpx.solution
    print("Solution status: ", solution.get_status())
    print("Objective value: ", solution.get_objective_value())

    # Extract y solution
    sol_y = []
    for s in range(data.num_s):
        vals = solution.get_values(y[s])
        sol_y.append(vals)

    print("y solution (first 5 elements):")
    for s in range(min(data.num_s,5)):
        print(sol_y[s][:min(data.num_d,5)])

    return solution.get_status(), solution.get_objective_value(), sol_y

# =====================
# Example Run
# =====================
if __name__ == "__main__":
    # Example data:
    # Suppose we have 2 supply nodes and 3 demand nodes
    # supply = [30, 40]
    # demand = [20, 25, 25] sum of demand = 70, sum of supply = 70
    # fixed costs f_sd and variable costs c_sd as small matrices:
    supply = [30,40]
    demand = [20,25,25]

    f = [
         [100,120,100],
         [80,  90,110]
        ]

    c = [
         [4,5,3],
         [2,6,9]
        ]

    # decompose_fractional = True means also separate fractional solutions
    decompose_fractional = True
    fctp_benders(decompose_fractional, supply, demand, f, c)

