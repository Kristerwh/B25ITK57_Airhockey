# from mushroom_rl.core import Environment
#
#
# class AirHockeyChallengeWrapper(Environment):
#     def __init__(self, env, mdp_info=None, **kwargs):
#
#         env_dict = {
#             "3dof-hit": lambda: print(),
#             "3dof-defend": lambda: print(),
#         }
#
#         if mdp_info is None:
#             print()
#             self.mdp_info = type('MDPInfo', (), {"dt": 0.02})()  # Minimal placeholder
#         else:
#             self.mdp_info = mdp_info
#
#         self.base_env = env_dict[env]()