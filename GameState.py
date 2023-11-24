class GameState():
  def __init__(self, pyboy):
    game_wrapper = pyboy.game_wrapper()

    self.time_left = game_wrapper.time_left
    self.lives_left = game_wrapper.lives_left
    self.score = game_wrapper.score
    self.level_progress = game_wrapper.level_progress
    self.world = game_wrapper.world
    self._level_progress_max = game_wrapper._level_progress_max

    # level_progress variable seems to have a bug, but below code calculates it the x movement right

    # Code from https://github.com/lixado/PyBoy-RL/blob/main/AISettings/MarioAISettings.py#L10-L17
    level_block = pyboy.get_memory_value(0xC0AB)
    # C202 Mario's X position relative to the screen
    mario_x = pyboy.get_memory_value(0xC202)
    scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
    real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
    self.real_x_pos = level_block * 16 + real + mario_x
