class TemplateScore:

  def __init__(self, score, crop_image, template_rect, overlap_score, image_valid_points=None, option_used=None, template=None):
    self.score = score
    self.crop_image = crop_image
    self.image_valid_points=image_valid_points
    self.rect = template_rect
    self.overlap_score = overlap_score
    self.option = option_used
    self.template = template

  def __str__(self):
    return f'TemplateScore(ovlp_score:{self.overlap_score}, score={self.score}, rect={self.rect})'
