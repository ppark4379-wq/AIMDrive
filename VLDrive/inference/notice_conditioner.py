"""Add live route notices to the same model input contract used by training."""
from utils.notice_policy import NoticeTracker, RouteNoticePolicy


class NoticeConditioner:
    """One instance per live ego stream. Visual inputs are already batched [1,T,N,D]."""
    def __init__(self, zones=None):
        self.tracker = NoticeTracker(RouteNoticePolicy(zones) if zones else RouteNoticePolicy())

    def reset(self):
        self.tracker.reset()

    def model_inputs(self, visual_feature, navigation, x, y):
        state = self.tracker.update(x, y, navigation)
        inputs = {"visual_feature": visual_feature, "instruction": [navigation],
                  "notice_text": [state["notice_text"]]}
        return inputs, state

    def predict(self, model, visual_feature, navigation, x, y, **kwargs):
        inputs, state = self.model_inputs(visual_feature, navigation, x, y)
        return model.predict(inputs, **kwargs), state
