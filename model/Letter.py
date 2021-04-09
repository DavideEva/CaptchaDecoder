class Letter:
  def __init__(self, rectangle, letter, score, letter_image_position=None):
    self.rect = rectangle
    self.letter = letter
    self.score = score
    self.letter_image_position = letter_image_position

  def __lt__(self, other):
    return self.score < other.score

  def __repr__(self):
    return f'Letter({self.letter}, score={self.score})'

  def __str__(self):
    return f'Letter({self.letter}, score={self.score}, rect={self.rect})'
