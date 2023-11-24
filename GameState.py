class GameState():
  def __init__(self, pyboy):
    game_wrapper = pyboy.game_wrapper()

    self.time_left = game_wrapper.time_left
    self.lives_left = game_wrapper.lives_left
    self.score = game_wrapper.score
    self.level_progress = game_wrapper.level_progress
    self.world = game_wrapper.world
    self._level_progress_max = game_wrapper._level_progress_max