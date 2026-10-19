"""Pose-aware point-centric CDF objective.

Same GT metric-depth L1, objectness, graspness and view losses as the
unmodified EconomicGrasp point baseline. Only the final angle/depth/score/width
supervision is replaced with evaluator-aligned angle*depth CDF and depth-wise
width using the existing CVA-CDF loss functions (same normalization / scaling).
"""
from models.loss_economicgrasp_depth import (
    compute_depth_reg_loss,
    compute_objectness_loss,
    compute_graspness_loss,
    compute_view_graspness_loss,
)
from models.loss_economicgrasp_depth_kview_transformer import (
    compute_cva_cdf_loss,
    compute_cva_width_depth_loss,
)
from utils.arguments import cfgs


def get_point_cdf_loss(end_points):
    if "grasp_cdf_pred_angle_depth" not in end_points:
        raise KeyError("Missing point CDF head outputs")
    depth_loss, end_points=compute_depth_reg_loss(end_points)
    objectness, end_points=compute_objectness_loss(end_points)
    graspness, end_points=compute_graspness_loss(end_points)
    view, end_points=compute_view_graspness_loss(end_points)
    # Use the SAME unbalanced CDF BCE as the official CVA-CDF baseline.
    cdf, end_points=compute_cva_cdf_loss(end_points,balanced=False)
    width, end_points=compute_cva_width_depth_loss(end_points)

    grasp_loss=(
        cfgs.graspness_loss_weight*graspness
        +cfgs.view_loss_weight*view
        +cfgs.score_loss_weight*cdf
        +cfgs.width_loss_weight*width
    )
    overall=(
        cfgs.objectness_loss_weight*objectness
        +cfgs.depth_prob_loss_weight*depth_loss
        +grasp_loss
    )
    end_points["A: Objectness Loss"]=objectness
    end_points["A: Grasp Loss"]=grasp_loss
    end_points["A: DepthReg Loss"]=cfgs.depth_prob_loss_weight*depth_loss
    end_points["A: Overall Loss"]=overall
    return overall,end_points
