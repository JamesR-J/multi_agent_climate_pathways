import logging
import time

import torch
import numpy as np
import yaml
from gym import Env
import os
import sys


class RiceN(Env):
    def __init__(self, discount=0.99, t0=0, dt=1, reward_type='PB', max_steps=600, image_dir='./images/', run_number=0,
                 plot_progress=False, num_agents=2, **kwargs):
        self.small_num = 1e-0

        self.num_agents = num_agents

        self.num_discrete_action_levels = 9  # 2  # 10
        self.negotiation_on = False  # negotiation_on
        self.float_dtype = np.float32
        self.int_dtype = np.int32

        self.savings_action_nvec = [self.num_discrete_action_levels]
        self.mitigation_rate_action_nvec = [self.num_discrete_action_levels]
        # Each region sets max allowed export from own region
        self.export_action_nvec = [self.num_discrete_action_levels]
        # Each region sets import bids (max desired imports from other countries)
        self.import_actions_nvec = [self.num_discrete_action_levels] * self.num_agents
        # Each region sets import tariffs imposed on other countries
        self.tariff_actions_nvec = [self.num_discrete_action_levels] * self.num_agents

        self.actions_nvec = (
                self.savings_action_nvec
                + self.mitigation_rate_action_nvec
                + self.export_action_nvec
                + self.import_actions_nvec
                + self.tariff_actions_nvec
        )

        self.management_cost = 0.5
        self.image_dir = image_dir
        self.run_number = run_number
        self.plot_progress = plot_progress
        self.max_steps = max_steps
        self.gamma = discount

        params = self.set_rice_params(os.path.join('./envs/', "region_yamls/"))
        self.rice_constant = params["_RICE_CONSTANT"]
        self.dice_constant = params["_DICE_CONSTANT"]
        self.all_constants = self.concatenate_world_and_regional_params(self.dice_constant, self.rice_constant)

        self.balance_interest_rate = 0.1

        # Parameters for Armington aggregation
        # TODO : add to yaml
        self.sub_rate = 0.5
        self.dom_pref = 0.5
        if self.num_agents > 1:
            self.for_pref = [0.5 / (self.num_agents - 1)] * self.num_agents
        else:
            self.for_pref = [1.0]  # TODO check this is the right value

        # Typecasting
        self.sub_rate = np.array([self.sub_rate]).astype(self.float_dtype)
        self.dom_pref = np.array([self.dom_pref]).astype(self.float_dtype)
        self.for_pref = np.array(self.for_pref, dtype=self.float_dtype)

        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)
        self.reward = torch.tensor([0.0]).repeat(self.num_agents, 1)
        self.old_emissions = torch.tensor([0.0]).repeat(self.num_agents, 1)

        self.reward_type = reward_type
        print("This is the reward type: {}".format(reward_type))

        timeStart = 0
        intSteps = 10  # integration Steps
        self.current_year = None
        self.t = self.t0 = t0
        self.dt = dt
        self.next_t = self.t + self.dt

        self.sim_time_step = np.linspace(timeStart, dt, intSteps)

        # Definitions from outside
        # self.current_state = torch.tensor([0.0, 0.0, 0.0]).repeat(self.num_agents, 1)
        # self.state = self.start_state = self.current_state
        self.state = torch.tensor([0.0, 0.0, 0.0]).repeat(self.num_agents, 1)
        self.observation_space = torch.tensor([0.0]*(self.num_agents + (7 * self.num_agents) + 13)).repeat(self.num_agents, 1)  # TODO need to confirm this stuff

        x, y = torch.meshgrid(torch.arange(10), torch.arange(10), indexing='ij')
        self.action_space = torch.stack((x.flatten(), y.flatten()),dim=1)  # .unsqueeze(0).expand(self.num_agents, -1, -1)

        self.temp_pb = torch.tensor([7.0]).repeat(self.num_agents, 1)  # TODO need to find a better num with backup paper
        self.Y_SF = torch.tensor([0.0]).repeat(self.num_agents, 1)
        self.S_LIMIT = torch.tensor(torch.inf).repeat(self.num_agents, 1)  # TODO this causes scaling issues so not good to use
        self.PB = torch.cat((self.S_LIMIT, self.Y_SF, self.temp_pb), dim=1)

        self.global_state = {}

        self.old_norm = torch.tensor([0.0]).repeat(self.num_agents, 1)

        # self.PB = torch.cat((self.A_PB, self.Y_SF, self.S_LIMIT), dim=1)

    def read_yaml_data(self, yaml_file):
        """Helper function to read yaml configuration data."""
        with open(yaml_file, "r", encoding="utf-8") as file_ptr:
            file_data = file_ptr.read()
        file_ptr.close()
        data = yaml.load(file_data, Loader=yaml.FullLoader)
        return data

    def set_rice_params(self, yamls_folder=None):
        """Helper function to read yaml data and set environment configs."""
        assert yamls_folder is not None
        dice_params = self.read_yaml_data(os.path.join(yamls_folder, "default.yml"))
        file_list = sorted(os.listdir(yamls_folder))  #
        yaml_files = []
        for file in file_list:
            if file[-4:] == ".yml" and file != "default.yml":
                yaml_files.append(file)

        rice_params = []
        for file in yaml_files[0:self.num_agents]:
            rice_params.append(self.read_yaml_data(os.path.join(yamls_folder, file)))

        # Overwrite rice params
        num_regions = len(rice_params)
        for k in dice_params["_RICE_CONSTANT"].keys():
            dice_params["_RICE_CONSTANT"][k] = [
                                                   dice_params["_RICE_CONSTANT"][k]
                                               ] * num_regions
        for idx, param in enumerate(rice_params):
            for k in param["_RICE_CONSTANT"].keys():
                dice_params["_RICE_CONSTANT"][k][idx] = param["_RICE_CONSTANT"][k]

        return dice_params

    def concatenate_world_and_regional_params(self, world, regional):
        """
        This function merges the world params dict into the regional params dict.
        Inputs:
            world: global params, dict, each value is common to all regions.
            regional: region-specific params, dict,
                      length of the values should equal the num of regions.
        Outputs:
            outs: list of dicts, each dict corresponding to a region
                  and comprises the global and region-specific parameters.
        """
        vals = regional.values()
        assert all(
            len(item) == self.num_agents for item in vals
        ), "The number of regions has to be consistent!"

        outs = []
        for region_id in range(self.num_agents):
            out = world.copy()
            for key, val in regional.items():
                out[key] = val[region_id]
            outs.append(out)
        return outs

    def reset(self):
        """
        Reset the environment
        """
        self.t = 0
        self.next_t = 0
        self.current_year = self.all_constants[0]["xt_0"]
        self.reward = torch.tensor([0.0]).repeat(self.num_agents, 1)
        self.final_state = torch.tensor([False]).repeat(self.num_agents, 1)

        constants = self.all_constants

        self.set_global_state(
            key="global_temperature",
            value=np.array([constants[0]["xT_AT_0"], constants[0]["xT_LO_0"]]),
            timestep=self.t,
            norm=1e1,
        )

        self.set_global_state(
            key="global_carbon_mass",
            value=np.array(
                [constants[0]["xM_AT_0"],
                 constants[0]["xM_UP_0"],
                 constants[0]["xM_LO_0"]]),
            timestep=self.t,
            norm=1e4,
        )

        self.set_global_state(
            key="capital_all_regions",
            value=np.array([constants[region_id]["xK_0"] for region_id in range(self.num_agents)]),
            timestep=self.t,
            norm=1e4,
        )

        self.set_global_state(
            key="labor_all_regions",
            value=np.array([constants[region_id]["xL_0"] for region_id in range(self.num_agents)]),
            timestep=self.t,
            norm=1e4,
        )

        self.set_global_state(
            key="production_factor_all_regions",
            value=np.array([constants[region_id]["xA_0"] for region_id in range(self.num_agents)]),
            timestep=self.t,
            norm=1e2,
        )

        self.set_global_state(
            key="intensity_all_regions",
            value=np.array([constants[region_id]["xsigma_0"] for region_id in range(self.num_agents)]),
            timestep=self.t,
            norm=1e-1,
        )

        for key in ["global_exogenous_emissions", "global_land_emissions", ]:
            self.set_global_state(
                key=key,
                value=np.zeros(1, ),
                timestep=self.t,
            )
        self.set_global_state("timestep", self.t, self.t, dtype=self.int_dtype, norm=1e2)
        self.set_global_state(
            "activity_timestep",
            self.next_t,
            self.t,
            dtype=self.int_dtype,
        )

        for key in [
            "capital_depreciation_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "max_export_limit_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(self.num_agents),
                timestep=self.t,
            )

        for key in [
            "consumption_all_regions",
            "current_balance_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "production_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(self.num_agents),
                timestep=self.t,
                norm=1e3,
            )

        for key in [
            "tariffs",
            "future_tariffs",
            "scaled_imports",
            "desired_imports",
            "tariffed_imports",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros((self.num_agents, self.num_agents)),
                timestep=self.t,
                norm=1e2,
            )

        # Negotiation-related features
        self.set_global_state(
            key="stage",
            value=np.zeros(1),
            timestep=self.t,
            dtype=self.int_dtype,
        )
        self.set_global_state(
            key="minimum_mitigation_rate_all_regions",
            value=np.zeros(self.num_agents),
            timestep=self.t,
        )
        for key in [
            "promised_mitigation_rate",
            "requested_mitigation_rate",
            "proposal_decisions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros((self.num_agents, self.num_agents)),
                timestep=self.t,
            )

        global_temperature = self.get_global_state("global_temperature", self.t)

        result = torch.tensor([0.0, 0.0, 0.0]).repeat(self.num_agents, 1)
        result[:, 2] = torch.tensor(global_temperature[0])

        for agent in range(self.num_agents):
            intensity = self.get_global_state("intensity_all_regions", timestep=self.t-1, region_id=agent)
            mitigation_rate = self.get_global_state("mitigation_rate_all_regions", region_id=agent)
            production = self.get_global_state("production_all_regions", region_id=agent)
            land_emissions = self.get_global_state("global_land_emissions")

            result[agent, 0] = torch.tensor(self.get_aux_m(intensity, mitigation_rate, production, land_emissions))
            result[agent, 1] = torch.tensor(self.get_global_state("capital_all_regions", timestep=self.t, region_id=agent))

        self.state = result
        self.old_emissions = self.state[:, 0]

        return self.state, self.generate_observation()

    def action_split(self, action_vec):
        first_digit = action_vec // 10
        second_digit = action_vec % 10

        return torch.stack((first_digit, second_digit), dim=1)

    def generate_observation(self):
        """
        Generate observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            # "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            # "consumption_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            # "max_export_limit_all_regions",
            # "current_balance_all_regions",
            # "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            # "utility_all_regions",
            # "social_welfare_all_regions",
            # "reward_all_regions",
        ]

        shared_features = np.array([])
        for feature in global_features + public_features:
            shared_features = np.append(shared_features, self.flatten_array(
                    self.global_state[feature]["value"][self.t] / self.global_state[feature]["norm"]
                    ),
            )

        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for agent in range(self.num_agents):
            # Add a region indicator array to the observation
            region_indicator = np.zeros(self.num_agents, dtype=self.float_dtype)
            region_indicator[agent] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_agents
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][self.t, agent]
                        / self.global_state[feature]["norm"]
                    ),
                )

            # for feature in bilateral_features:
            #     assert self.global_state[feature]["value"].shape[1] == self.num_regions
            #     assert self.global_state[feature]["value"].shape[2] == self.num_regions
            #     all_features = np.append(
            #         all_features,
            #         self.flatten_array(
            #             self.global_state[feature]["value"][self.timestep, region_id]
            #             / self.global_state[feature]["norm"]
            #         ),
            #     )
            #     all_features = np.append(
            #         all_features,
            #         self.flatten_array(
            #             self.global_state[feature]["value"][self.timestep, :, region_id]
            #             / self.global_state[feature]["norm"]
            #         ),
            #     )

            features_dict[agent] = all_features

        # Form the observation dictionary keyed by region id.
        obs_dict = {}
        for agent in range(self.num_agents):
            obs_dict[agent] = {
                "features": features_dict[agent],
            }

        tensor_list = [torch.tensor(value['features']) for value in obs_dict.values()]
        obs_tensor = torch.stack(tensor_list)

        return obs_tensor

    def step(self, action):
        self.t += 1

        for key in self.global_state:
            if key != "reward_all_regions":
                self.global_state[key]["value"][self.t] = self.global_state[key]["value"][self.t - 1].copy()

        self.set_global_state("timestep", self.t, self.t, dtype=self.int_dtype)
        # self.next_t = self.t + self.dt  # TODO check that making next_t a self function actually updates

        self.state = self._perform_step(action)
        self.observation_space = self.generate_observation()

        print(self.state)
        if not self.final_state.bool().any():
            assert torch.all(self.state[:, 2] == self.state[0, 2]), "Values in the first column are not all equal"

        # self.t = self.next_t  # TODO not sure this is needed

        self.get_reward_function()

        for agent in range(self.num_agents):
            #     if self._arrived_at_final_state(agent):
            #         self.final_state[agent] = True
            if not self._inside_planetary_boundaries(agent):
                self.final_state[agent] = True
            if self.final_state[agent]:
                self.reward[agent] += self.calculate_expected_final_reward(agent)
        #     if self.final_state[agent] and self.reward_type == "PB_ext":  # TODO means can get the final step if done ie the extra or less reward for PB_ext - bit of a dodgy workaround may look at altering the reward placement in the step function
        #         self.get_reward_function()

        # print(self.state)
        # print(self.reward)

        self.old_emissions = self.state[:, 0]

        return self.state, self.reward, self.final_state, self.observation_space

    def _perform_step(self, action):
        self.next_t += 1  # TODO dodgy fix until figure the issue

        self.set_global_state(key="activity_timestep", value=self.next_t, timestep=self.t, dtype=self.int_dtype)

        assert self.t == self.next_t

        action_n = self.action_split(action)

        assert len(action_n) == self.num_agents

        # add actions to global state
        savings_action_index = 0
        mitigation_rate_action_index = savings_action_index + len(self.savings_action_nvec)
        # export_action_index = mitigation_rate_action_index + len(self.mitigation_rate_action_nvec)
        # tariffs_action_index = export_action_index + len(self.export_action_nvec)
        # desired_imports_action_index = tariffs_action_index + len(self.tariff_actions_nvec)

        self.set_global_state("savings_all_regions",
                              [
                                  action_n[agent][savings_action_index].item() / self.num_discrete_action_levels for
                                  agent in range(self.num_agents)
                              ],
                              self.t)
        self.set_global_state("mitigation_rate_all_regions",
                              [
                                  action_n[agent][mitigation_rate_action_index].item() / self.num_discrete_action_levels
                                  for agent in range(self.num_agents)
                              ],
                              self.t)
        # self.set_global_state("max_export_limit_all_regions",
        #                       [
        #                           action[agent][export_action_index] / self.num_discrete_action_levels
        #                           for agent in range(self.num_agents)
        #                       ],
        #                       self.t)
        # self.set_global_state("future_tariffs",
        #                       [
        #                           action[agent][tariffs_action_index: tariffs_action_index + self.num_agents]
        #                           / self.num_discrete_action_levels
        #                           for agent in range(self.num_agents)
        #                       ],
        #                       self.t,
        #                       )
        # self.set_global_state("desired_imports",
        #                       [
        #                           action[agent][desired_imports_action_index: desired_imports_action_index + self.num_agents]
        #                           / self.num_discrete_action_levels
        #                           for agent in range(self.num_agents)
        #                       ],
        #                       self.t,
        #                       )

        # Constants
        constants = self.all_constants

        const = constants[0]
        aux_m_all_regions = np.zeros(self.num_agents, dtype=self.float_dtype)

        prev_global_temperature = self.get_global_state("global_temperature", self.t - 1)
        t_at = prev_global_temperature[0]

        # add emissions to global state
        global_exogenous_emissions = self.get_exogenous_emissions(const["xf_0"], const["xf_1"], self.max_steps,  # const["xt_f"],
                                                                  self.next_t)
        global_land_emissions = self.get_land_emissions(const["xE_L0"], const["xdelta_EL"], self.next_t,
                                                        self.num_agents)

        self.set_global_state("global_exogenous_emissions", global_exogenous_emissions, self.t)
        self.set_global_state("global_land_emissions", global_land_emissions, self.t)
        # desired_imports = self.get_global_state("desired_imports")
        # scaled_imports = self.get_global_state("scaled_imports")

        for agent in range(self.num_agents):
            # Actions
            savings = self.get_global_state("savings_all_regions", region_id=agent)
            mitigation_rate = self.get_global_state("mitigation_rate_all_regions", region_id=agent)

            # feature values from previous timestep
            intensity = self.get_global_state("intensity_all_regions", timestep=self.t - 1, region_id=agent)
            production_factor = self.get_global_state("production_factor_all_regions", timestep=self.t - 1,
                                                      region_id=agent)
            capital = self.get_global_state("capital_all_regions", timestep=self.t - 1, region_id=agent)
            labor = self.get_global_state("labor_all_regions", timestep=self.t - 1, region_id=agent)
            # gov_balance_prev = self.get_global_state("current_balance_all_regions", timestep=self.t - 1,
            #                                          region_id=agent)

            # constants
            const = constants[agent]

            # climate costs and damages
            mitigation_cost = self.get_mitigation_cost(const["xp_b"], const["xtheta_2"], const["xdelta_pb"],
                                                       self.next_t, intensity)

            damages = self.get_damages(t_at, const["xa_1"], const["xa_2"], const["xa_3"])
            abatement_cost = self.get_abatement_cost(mitigation_rate, mitigation_cost, const["xtheta_2"])
            production = self.get_production(production_factor, capital, labor, const["xgamma"])

            gross_output = self.get_gross_output(damages, abatement_cost, production)
            # gov_balance_prev = gov_balance_prev * (1 + self.balance_interest_rate)
            investment = self.get_investment(savings, gross_output)

            # for j in range(self.num_agents):
            #     scaled_imports[agent][j] = (desired_imports[agent][j] * gross_output)
            # # Import bid to self is reset to zero
            # scaled_imports[agent][agent] = 0
            #
            # total_scaled_imports = np.sum(scaled_imports[agent])
            # if total_scaled_imports > gross_output:
            #     for j in range(self.num_agents):
            #         scaled_imports[agent][j] = (scaled_imports[agent][j] / total_scaled_imports * gross_output)

            # # Scale imports based on gov balance
            # init_capital_multiplier = 10.0
            # debt_ratio = gov_balance_prev / init_capital_multiplier * const["xK_0"]
            # debt_ratio = min(0.0, debt_ratio)
            # debt_ratio = max(-1.0, debt_ratio)
            # debt_ratio = np.array(debt_ratio).astype(self.float_dtype)
            # scaled_imports[agent] *= 1 + debt_ratio

            self.set_global_state("mitigation_cost_all_regions", mitigation_cost, self.t, region_id=agent)
            self.set_global_state("damages_all_regions", damages, self.t, region_id=agent)
            self.set_global_state("abatement_cost_all_regions", abatement_cost, self.t, region_id=agent)
            self.set_global_state("production_all_regions", production, self.t, region_id=agent)
            self.set_global_state("gross_output_all_regions", gross_output, self.t, region_id=agent)
            # self.set_global_state("current_balance_all_regions", gov_balance_prev, self.t, region_id=agent)
            self.set_global_state("investment_all_regions", investment, self.t, region_id=agent)

        # for agent in range(self.num_agents):
        #     x_max = self.get_global_state("max_export_limit_all_regions", region_id=agent)
        #     gross_output = self.get_global_state("gross_output_all_regions", region_id=agent)
        #     investment = self.get_global_state("investment_all_regions", region_id=agent)
        #
        #     # scale desired imports according to max exports
        #     max_potential_exports = self.get_max_potential_exports(x_max, gross_output, investment)
        #     total_desired_exports = np.sum(scaled_imports[:, agent])
        #
        #     if total_desired_exports > max_potential_exports:
        #         for j in range(self.num_agents):
        #             scaled_imports[j][agent] = (
        #                     scaled_imports[j][agent] / total_desired_exports * max_potential_exports)
        #
        # self.set_global_state("scaled_imports", scaled_imports, self.t)

        # # countries with negative gross output cannot import
        # prev_tariffs = self.get_global_state("future_tariffs", timestep=self.t - 1)
        # tariffed_imports = self.get_global_state("tariffed_imports")
        # scaled_imports = self.get_global_state("scaled_imports")

        # for agent in range(self.num_agents):
        #     # constants
        #     const = constants[agent]
        #
        #     # get variables from global state
        #     savings = self.get_global_state("savings_all_regions", region_id=agent)
        #     gross_output = self.get_global_state("gross_output_all_regions", region_id=agent)
        #     investment = self.get_investment(savings, gross_output)
        #     labor = self.get_global_state("labor_all_regions", timestep=self.t - 1, region_id=agent)
        #
        #     # calculate tariffed imports, tariff revenue and budget balance
        #     for j in range(self.num_agents):
        #         tariffed_imports[agent, j] = scaled_imports[agent, j] * (1 - prev_tariffs[agent, j])
        #     tariff_revenue = np.sum(scaled_imports[agent, :] * prev_tariffs[agent, :])
        #
        #     # Aggregate consumption from domestic and foreign goods
        #     # domestic consumption
        #     c_dom = self.get_consumption(gross_output, investment, exports=scaled_imports[:, agent])
        #
        #     consumption = self.get_armington_agg(c_dom=c_dom,
        #                                          c_for=tariffed_imports[agent, :],  # np.array
        #                                          sub_rate=self.sub_rate,  # in (0,1)  np.array
        #                                          dom_pref=self.dom_pref,  # in [0,1]  np.array
        #                                          for_pref=self.for_pref,  # np.array, sums to (1 - dom_pref)
        #                                          )
        #
        #     utility = self.get_utility(labor, consumption, const["xalpha"])
        #
        #     social_welfare = self.get_social_welfare(utility, const["xrho"], const["xDelta"], self.next_t)
        #
        #     self.set_global_state("tariff_revenue", tariff_revenue, self.t, region_id=agent)
        #     self.set_global_state("consumption_all_regions", consumption, self.t, region_id=agent)
        #     self.set_global_state("utility_all_regions", utility, self.t, region_id=agent)
        #     self.set_global_state("social_welfare_all_regions", social_welfare, self.t, region_id=agent)
        #     self.set_global_state("reward_all_regions", utility, self.t, region_id=agent)

        # # Update gov balance
        # for agent in range(self.num_agents):
        #     const = constants[agent]
        #     gov_balance_prev = self.get_global_state("current_balance_all_regions", region_id=agent)
        #     scaled_imports = self.get_global_state("scaled_imports")
        #
        #     gov_balance = gov_balance_prev + const["xDelta"] * (
        #             np.sum(scaled_imports[:, agent]) - np.sum(scaled_imports[agent, :]))
        #     self.set_global_state("current_balance_all_regions", gov_balance, self.t, region_id=agent)
        #
        # self.set_global_state("tariffed_imports", tariffed_imports, self.t)

        # Update temperature
        m_at = self.get_global_state("global_carbon_mass", timestep=self.t - 1)[0]
        prev_global_temperature = self.get_global_state("global_temperature", timestep=self.t - 1)

        global_exogenous_emissions = self.get_global_state("global_exogenous_emissions")[0]

        const = constants[0]
        global_temperature = self.get_global_temperature(np.array(const["xPhi_T"]),
                                                         prev_global_temperature,
                                                         const["xB_T"],
                                                         const["xF_2x"],
                                                         m_at,
                                                         const["xM_AT_1750"],
                                                         global_exogenous_emissions,
                                                         )
        self.set_global_state("global_temperature", global_temperature, self.t)

        for agent in range(self.num_agents):
            intensity = self.get_global_state("intensity_all_regions", timestep=self.t - 1, region_id=agent)
            mitigation_rate = self.get_global_state("mitigation_rate_all_regions", region_id=agent)
            production = self.get_global_state("production_all_regions", region_id=agent)
            land_emissions = self.get_global_state("global_land_emissions")

            aux_m = self.get_aux_m(intensity, mitigation_rate, production, land_emissions)
            aux_m_all_regions[agent] = aux_m

        # Update carbon mass
        const = constants[0]
        prev_global_carbon_mass = self.get_global_state("global_carbon_mass", timestep=self.t - 1)
        global_carbon_mass = self.get_global_carbon_mass(const["xPhi_M"], prev_global_carbon_mass, const["xB_M"],
                                                         np.sum(aux_m_all_regions))
        self.set_global_state("global_carbon_mass", global_carbon_mass, self.t)

        result = torch.tensor([0.0, 0.0, 0.0]).repeat(self.num_agents, 1)
        result[:, 0] = torch.tensor(aux_m_all_regions)
        result[:, 2] = global_temperature[0]

        for agent in range(self.num_agents):
            capital = self.get_global_state("capital_all_regions", timestep=self.t - 1, region_id=agent)
            labor = self.get_global_state("labor_all_regions", timestep=self.t - 1, region_id=agent)
            production_factor = self.get_global_state("production_factor_all_regions", timestep=self.t - 1,
                                                      region_id=agent)
            intensity = self.get_global_state("intensity_all_regions", timestep=self.t - 1, region_id=agent)
            investment = self.get_global_state("investment_all_regions", timestep=self.t, region_id=agent)

            const = constants[agent]

            capital_depreciation = self.get_capital_depreciation(const["xdelta_K"], const["xDelta"])
            updated_capital = self.get_capital(capital_depreciation, capital, const["xDelta"], investment)
            updated_capital = updated_capital

            updated_labor = self.get_labor(labor, const["xL_a"], const["xl_g"])
            updated_production_factor = self.get_production_factor(production_factor,
                                                                   const["xg_A"],
                                                                   const["xdelta_A"],
                                                                   const["xDelta"],
                                                                   self.next_t,
                                                                   )
            updated_intensity = self.get_carbon_intensity(intensity,
                                                          const["xg_sigma"],
                                                          const["xdelta_sigma"],
                                                          const["xDelta"],
                                                          self.next_t,
                                                          )

            self.set_global_state("capital_depreciation_all_regions", capital_depreciation, self.t)
            self.set_global_state("capital_all_regions", updated_capital, self.t, region_id=agent)
            self.set_global_state("labor_all_regions", updated_labor, self.t, region_id=agent)
            self.set_global_state("production_factor_all_regions", updated_production_factor, self.t, region_id=agent)
            self.set_global_state("intensity_all_regions", updated_intensity, self.t, region_id=agent)

            result[agent, 1] = updated_capital

        # self.set_global_state("tariffs", self.global_state["future_tariffs"]["value"][self.t], self.t)

        # obs = self.generate_observation()
        # rew = {
        #     region_id: self.global_state["reward_all_regions"]["value"][
        #         self.timestep, region_id
        #     ]
        #     for region_id in range(self.num_regions)
        # }
        # # Set current year
        self.current_year += self.all_constants[0]["xDelta"]

        return result

    def get_reward_function(self):
        def reward_distance_PB(agent, action=0):
            self.reward[agent] = 0.0

            if self._inside_planetary_boundaries(agent):
                self.reward[agent] = torch.norm(self.state[agent, 2] - self.temp_pb[agent])
            else:
                self.reward[agent] = -1.0

        def posi_negi_PB(agent, action=0):
            self.reward[agent] = 0.0

            if self._inside_planetary_boundaries(agent):
                new_norm = torch.norm(self.state[agent, 2] - self.temp_pb[agent])
                reward = torch.abs(new_norm - self.old_norm) * 10
                if self.t <= 1:
                    self.old_norm = new_norm

                if self.old_norm > new_norm:
                    reward = -reward
                elif self.old_norm < new_norm:
                    reward = reward
                elif self.old_norm == new_norm:
                    reward = torch.tensor(0.0)
                else:
                    print("FAILURE")
                    sys.exit(1)

                self.reward[agent] = reward

                self.old_norm = new_norm
            else:
                self.reward[agent] = -10.0

        def posi_negi_carbon(agent, action=0):
            reward = 0.0  # TODO maybe make it just a minimise to zero kinda thing
            old_carbon = self.get_global_state("global_carbon_mass", timestep=self.t - 1)[0]
            new_carbon = self.get_global_state("global_carbon_mass", timestep=self.t)[0]

            reward = abs(old_carbon - new_carbon)  # ** 2

            if old_carbon > new_carbon:
                reward = reward
            elif old_carbon < new_carbon:
                reward = -reward
            elif old_carbon == new_carbon:
                reward = 0.0
            else:
                print("FAILURE")
                sys.exit(1)

            self.reward[agent] = torch.tensor(reward)

        def posi_negi_emissions(agent, action=0):
            reward = 0.0
            old_emissions = self.old_emissions[agent]
            new_emissions = self.state[agent, 0]

            reward = abs(old_emissions - new_emissions)

            if old_emissions > new_emissions:
                reward = reward
            elif old_emissions < new_emissions:
                reward = -reward
            elif old_emissions == new_emissions:
                reward = 0.0
            else:
                print("FAILURE")
                sys.exit(1)

            self.reward[agent] = reward

        def PB_and_capital(agent, action=0):
            self.reward[agent] = 0.0

            if self._inside_planetary_boundaries(agent):
                self.reward[agent] += torch.norm(self.state[agent, 2] - self.PB[agent, 2])
                self.reward[agent] += torch.norm(self.state[agent, 1] - self.PB[agent, 1]) / 100
            else:
                self.reward[agent] = -10.0

        for agent in range(self.num_agents):
            if self.reward_type[agent] == 'PB':
                reward_distance_PB(agent)
            elif self.reward_type[agent] == 'posi_negi_PB':
                posi_negi_PB(agent)
            elif self.reward_type[agent] == 'cap_PB':
                PB_and_capital(agent)
            elif self.reward_type[agent] == 'carbon_reduc':
                posi_negi_carbon(agent)
            elif self.reward_type[agent] == 'emission_reduc':
                posi_negi_emissions(agent)
            elif self.reward_type[agent] == None:
                print("ERROR! You have to choose a reward function!\n",
                      "Available Reward functions for this environment are: PB, rel_share, survive, desirable_region!")
                exit(1)
            else:
                print("ERROR! The reward function you chose is not available! " + self.reward_type[agent])
                # print_debug_info()
                sys.exit(1)

    def calculate_expected_final_reward(self, agent):
        """
        Get the reward in the last state, expecting from now on always default.
        This is important since we break up simulation at final state, but we do not want the agent to
        find trajectories that are close (!) to final state and stay there, since this would
        result in a higher total reward.
        """
        remaining_steps = self.max_steps - self.t
        discounted_future_reward = 0.
        for i in range(remaining_steps):
            discounted_future_reward += self.gamma ** i * self.reward[agent]  # TODO works well unless agent that turns red comes back into the PB as it gives +ve numbher, but the agent doesnt update so it is fine

        # print("Agent number : {}".format(agent))
        # print(discounted_future_reward)
        return discounted_future_reward

    def set_global_state(self, key: object = None, value: object = None, timestep: object = None, norm: object = None,
                         region_id: object = None, dtype: object = None) -> object:
        """
        Set a specific slice of the environment global state with a key and value pair.
        The value is set for a specific timestep, and optionally, a specific region_id.
        Optionally, a normalization factor (used for generating observation),
        and a datatype may also be provided.
        """
        assert key is not None
        assert value is not None
        assert timestep is not None
        if norm is None:
            norm = 1.0
        if dtype is None:
            dtype = self.float_dtype

        if isinstance(value, list):
            value = np.array(value, dtype=dtype)
        elif isinstance(value, (float, np.floating)):
            value = np.array([value], dtype=self.float_dtype)
        elif isinstance(value, (int, np.integer)):
            value = np.array([value], dtype=self.int_dtype)
        else:
            assert isinstance(value, np.ndarray)

        if key not in self.global_state:
            logging.info("Adding %s to global state.", key)
            if region_id is None:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.max_steps + 1,) + value.shape, dtype=dtype
                    ),
                    "norm": norm,
                }
            else:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.max_steps + 1,) + (self.num_agents,) + value.shape,
                        dtype=dtype,
                    ),
                    "norm": norm,
                }

        # Set the value
        if region_id is None:
            self.global_state[key]["value"][timestep] = value
        else:
            self.global_state[key]["value"][timestep, region_id] = value

    def get_global_state(self, key=None, timestep=None, region_id=None):
        assert key in self.global_state, f"Invalid key '{key}' in global state!"
        if timestep is None:
            timestep = self.t
        if region_id is None:
            return self.global_state[key]["value"][timestep].copy()
        return self.global_state[key]["value"][timestep, region_id].copy()

    def _inside_planetary_boundaries(self, agent):  # TODO confirm this is correct
        e = self.state[agent, 0]
        y = self.state[agent, 1]
        global_temp_change = self.state[agent, 2]
        is_inside = True

        # if global_temp_change > self.A_PB[agent] or y < self.Y_SF[agent] or e < self.S_LIMIT[agent]:
        if global_temp_change > self.temp_pb[agent]:
            is_inside = False
            # print("Outside PB!")
        return is_inside

    @staticmethod
    def flatten_array(array):
        """Flatten a numpy array"""
        return np.reshape(array, -1)

    # RICE dynamics
    @staticmethod
    def get_mitigation_cost(p_b, theta_2, delta_pb, timestep, intensity):
        """Obtain the cost for mitigation."""
        return p_b / (1000 * theta_2) * pow(1 - delta_pb, timestep - 1) * intensity

    @staticmethod
    def get_exogenous_emissions(f_0, f_1, t_f, timestep):
        """Obtain the amount of exogeneous emissions."""
        return f_0 + min(f_1 - f_0, (f_1 - f_0) / t_f * (timestep - 1))

    @staticmethod
    def get_land_emissions(e_l0, delta_el, timestep, num_regions):
        """Obtain the amount of land emissions."""
        return e_l0 * pow(1 - delta_el, timestep - 1) / num_regions

    @staticmethod
    def get_production(production_factor, capital, labor, gamma):
        """Obtain the amount of goods produced."""
        return production_factor * pow(capital, gamma) * pow(labor / 1000, 1 - gamma)

    @staticmethod
    def get_damages(t_at, a_1, a_2, a_3):
        """Obtain damages."""
        return 1 / (1 + a_1 * t_at + a_2 * pow(t_at, a_3))

    @staticmethod
    def get_abatement_cost(mitigation_rate, mitigation_cost, theta_2):
        """Compute the abatement cost."""
        return mitigation_cost * pow(mitigation_rate, theta_2)

    @staticmethod
    def get_gross_output(damages, abatement_cost, production):
        """Compute the gross production output, taking into account
        damages and abatement cost."""
        return damages * (1 - abatement_cost) * production

    @staticmethod
    def get_investment(savings, gross_output):
        """Obtain the investment cost."""
        return savings * gross_output

    @staticmethod
    def get_consumption(gross_output, investment, exports):
        """Obtain the consumption cost."""
        total_exports = np.sum(exports)
        assert gross_output - investment - total_exports > -1e-5, "consumption cannot be negative!"
        return max(0.0, gross_output - investment - total_exports)

    @staticmethod
    def get_max_potential_exports(x_max, gross_output, investment):
        """Determine the maximum potential exports."""
        if x_max * gross_output <= gross_output - investment:
            return x_max * gross_output
        return gross_output - investment

    @staticmethod
    def get_capital_depreciation(x_delta_k, x_delta):
        """Compute the global capital depreciation."""
        return pow(1 - x_delta_k, x_delta)

    @staticmethod
    def get_global_temperature(phi_t, temperature, b_t, f_2x, m_at, m_at_1750, exogenous_emissions):
        """Get the temperature levels."""
        return np.dot(phi_t, temperature) + np.dot(
            b_t, f_2x * np.log(m_at / m_at_1750) / np.log(2) + exogenous_emissions
        )

    @staticmethod
    def get_aux_m(intensity, mitigation_rate, production, land_emissions):
        """Auxiliary variable to denote carbon mass levels."""
        return intensity * (1 - mitigation_rate) * production + land_emissions

    @staticmethod
    def get_global_carbon_mass(phi_m, carbon_mass, b_m, aux_m):
        """Get the carbon mass level."""
        return np.dot(phi_m, carbon_mass) + np.dot(b_m, aux_m)

    @staticmethod
    def get_capital(capital_depreciation, capital, delta, investment):
        """Evaluate capital."""
        return capital_depreciation * capital + delta * investment

    @staticmethod
    def get_labor(labor, l_a, l_g):
        """Compute total labor."""
        return labor * pow((1 + l_a) / (1 + labor), l_g)

    @staticmethod
    def get_production_factor(production_factor, g_a, delta_a, delta, timestep):
        """Compute the production factor."""
        return production_factor * (
                np.exp(0.0033) + g_a * np.exp(-delta_a * delta * (timestep - 1))
        )

    @staticmethod
    def get_carbon_intensity(intensity, g_sigma, delta_sigma, delta, timestep):
        """Determine the carbon emission intensity."""
        return intensity * np.exp(
            -g_sigma * pow(1 - delta_sigma, delta * (timestep - 1)) * delta
        )

    def get_utility(self, labor, consumption, alpha):
        """Obtain the utility."""
        return (
                (labor / 1000.0)
                * (pow(consumption / (labor / 1000.0) + self.small_num, 1 - alpha) - 1)
                / (1 - alpha)
        )

    @staticmethod
    def get_social_welfare(utility, rho, delta, timestep):
        """Compute social welfare"""
        return utility / pow(1 + rho, delta * timestep)

    @staticmethod
    def get_armington_agg(
            c_dom,
            c_for,  # np.array
            sub_rate=0.5,  # in (0,1)
            dom_pref=0.5,  # in [0,1]
            for_pref=None,  # np.array
    ):
        """
        Armington aggregate from Lessmann,2009.
        Consumption goods from different regions act as imperfect substitutes.
        As such, consumption of domestic and foreign goods are scaled according to
        relative preferences, as well as a substitution rate, which are modeled
        by a CES functional form.
        Inputs :
            `C_dom`     : A scalar representing domestic consumption. The value of
                        C_dom is what is left over from initial production after
                        investment and exports are deducted.
            `C_for`     : An array reprensenting foreign consumption. Each element
                        is the consumption imported from a given country.
            `sub_rate`  : A substitution parameter in (0,1). The elasticity of
                        substitution is 1 / (1 - sub_rate).
            `dom_pref`  : A scalar in [0,1] representing the relative preference for
                        domestic consumption over foreign consumption.
            `for_pref`  : An array of the same size as `C_for`. Each element is the
                        relative preference for foreign goods from that country.
        """

        c_dom_pref = dom_pref * (c_dom ** sub_rate)
        c_for_pref = np.sum(for_pref * pow(c_for, sub_rate))

        c_agg = (c_dom_pref + c_for_pref) ** (1 / sub_rate)  # CES function
        return c_agg


if __name__ == "__main__":
    ricen = RiceN()
    ricen.reset()
    action = {0: np.array([1, 5, 6, 4, 0, 2, 3, 8, 9, 8, 9, 1, 2, 1, 2, 0, 7, 3, 2, 8, 6, 1, 0, 6, 2, 7, 5, 6, 6, 8,
                           5, 5, 3, 6, 3, 2, 1, 3, 3, 8, 9, 2, 3, 0, 1, 8, 4, 0, 5, 1, 4, 8, 9, 2, 8, 5, 0]),
              1: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
                           5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 2: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 3: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 4: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 5: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 6: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 7: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 8: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 9: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 10: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 11: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 12: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 13: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 14: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 15: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 16: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 17: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 18: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 19: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 20: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 21: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 22: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 23: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 24: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 25: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              # 26: np.array([6, 5, 5, 3, 8, 1, 6, 9, 3, 9, 3, 6, 1, 6, 4, 7, 5, 2, 0, 0, 8, 5, 4, 8, 0, 4, 3, 4, 2, 8,
              #              5, 8, 1, 0, 0, 1, 2, 3, 0, 6, 7, 5, 2, 0, 4, 3, 9, 5, 2, 0, 4, 8, 7, 3, 4, 3, 1]),
              }
    for _ in range(10):
        ricen.step(action)
    print(ricen.global_state)
