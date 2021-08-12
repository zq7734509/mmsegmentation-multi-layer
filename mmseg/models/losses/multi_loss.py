import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import cv2
import numpy as np

def test(pred,
         label,
         theta0,
         theta):
    # pred = torch.softmax(pred, dim=1)  # 在channel维度上进行softmax,使得所有channel上所有对应像素点的取值相加之和为1
    # loss = torch.mean(pred)
    # loss += 1
    n, c, _, _ = pred.shape
    pred = torch.softmax(pred, dim=1)  # 在channel维度上进行softmax,使得所有channel上所有对应像素点的取值相加之和为1
    # one-hot vector of ground truth
    one_hot_gt = one_hot(label, c)  # 在channel维度上,gt的类别在对应的channel层中为1,在其他channel层中均为0
    pred_b = F.max_pool2d(
        1 - pred, kernel_size=3, stride=1, padding=(3 - 1) // 2)
    pred_b -= 1 - pred
    loss = torch.mean(one_hot_gt)
    return loss


def one_hot(label, n_classes, requires_grad=False):
    """Return One Hot Label"""
    # label0 = label[0,:,:].cpu().squeeze()
    # label1 = label[1,:,:].cpu().squeeze()
    #
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(label0)
    # plt.subplot(1, 2, 2)
    # plt.imshow(label1)
    #
    # plt.show()
    #label[label==255] = n_classes   # 将label中的ground-truth所对应的255先转化为5

    one_hot_label = torch.eye(
        256, requires_grad=requires_grad)[label]  #一个数字转换成一串比如5,共有10类则表示为[0,0,0,0,0,1,0,0,0,0]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)   #[8,10,224,224]
    one_hot_label_output = one_hot_label[:,:n_classes,:,:]

    # one_hot_label_output_sum = torch.sum(one_hot_label_output, dim=1)
    # one_like_tensor = torch.ones(one_hot_label_output_sum.shape)
    # zero_like_tensor = torch.zeros(one_hot_label_output_sum.shape)
    # one_hot_label_output_sum_normal = torch.where(one_hot_label_output_sum>0, one_like_tensor, zero_like_tensor)
    # one_hot_label_output_sum_normal = one_hot_label_output_sum_normal.view(2, -1)  # [1,10,50176]
    # aa = torch.sum(one_hot_label_output_sum_normal,1)

    return one_hot_label_output.cuda()


def boundary(count,
             pred,  # 此处为二分类
             label,
             theta0,
             theta):

    # show = True
    show = False
    n, c, _, _ = pred.shape
    pred = torch.softmax(pred, dim=1)  # 在channel维度上进行softmax,使得所有channel上所有对应像素点的取值相加之和为1

    c = 1
    pred = pred[:,0,:,:]
    pred = pred.unsqueeze(1)

    # one-hot vector of ground truth
    one_hot_gt = one_hot(label, 5)  # 在channel维度上,gt的类别在对应的channel层中为1,在其他channel层中均为0
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # boundary map
    gt_b = F.max_pool2d(
        1 - one_hot_gt, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b = gt_b - (1 - one_hot_gt)
    gt_b_channel_sum = torch.sum(gt_b, dim=1)
    one_like_tensor = torch.ones(gt_b_channel_sum.shape).cuda()
    zero_like_tensor = torch.zeros(gt_b_channel_sum.shape).cuda()
    gt_b_channel_sum_normal = torch.where(gt_b_channel_sum>0, one_like_tensor, zero_like_tensor)
    gt_b = gt_b_channel_sum_normal

    pred_b = pred
    # pred_b = F.max_pool2d(
    #     1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    # pred_b = pred_b - (1 - pred)

    if show:
        # make the boundary_heatmap
        ######################################              gt_b           ########################################################
        gt_b_map = gt_b.data.cpu().numpy()
        gt_b_map_0 = gt_b_map[0, :, :]
        gt_b_map_1 = gt_b_map[1, :, :]

        max_value = gt_b_map_0.max()
        min_value = gt_b_map_0.min()

        gt_b_map_0 = (gt_b_map_0 - min_value) / max_value
        gt_heatmap_0 = np.uint8(255 * gt_b_map_0)
        gt_heatmap_0 = cv2.applyColorMap(gt_heatmap_0, cv2.COLORMAP_HOT)

        index_str1 = str(count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_map/' + index_str1 + '_gt.png'
        cv2.imwrite(output_path_str1, gt_heatmap_0)


        max_value = gt_b_map_1.max()
        min_value = gt_b_map_1.min()

        gt_b_map_1 = (gt_b_map_1 - min_value) / max_value
        gt_heatmap_1 = np.uint8(255 * gt_b_map_1)
        gt_heatmap_1 = cv2.applyColorMap(gt_heatmap_1, cv2.COLORMAP_HOT)

        index_str2 = str(count+1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_map/' + index_str2 + '_gt.png'
        cv2.imwrite(output_path_str2, gt_heatmap_1)
        #
        # ##################################              pred_b          ####################################################
        pred_b_map = torch.sum(pred_b.data, dim=1)
        pred_b_map = pred_b_map.data.cpu().numpy()
        pred_b_map_0 = pred_b_map[0, :, :]
        pred_b_map_1 = pred_b_map[1, :, :]

        max_value = pred_b_map_0.max()
        min_value = pred_b_map_0.min()

        pred_b_map_0 = (pred_b_map_0 - min_value) / max_value
        pred_heatmap_0 = np.uint8(255 * pred_b_map_0)
        pred_heatmap_0 = cv2.applyColorMap(pred_heatmap_0, cv2.COLORMAP_HOT)

        index_str1 = str(count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_map/' + index_str1 + '_pred.png'
        cv2.imwrite(output_path_str1, pred_heatmap_0)


        max_value = pred_b_map_1.max()
        min_value = pred_b_map_1.min()

        pred_b_map_1 = (pred_b_map_1 - min_value) / max_value
        pred_heatmap_1 = np.uint8(255 * pred_b_map_1)
        pred_heatmap_1 = cv2.applyColorMap(pred_heatmap_1, cv2.COLORMAP_HOT)

        index_str2 = str(count+1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_map/' + index_str2 + '_pred.png'
        cv2.imwrite(output_path_str2, pred_heatmap_1)
        ##################################              end          ####################################################


    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    if show:
        #####################################              gt_b_ext           ########################################################
        gt_b_map = gt_b_ext.data.cpu().numpy()
        gt_b_map_0 = gt_b_map[0, :, :]
        gt_b_map_1 = gt_b_map[1, :, :]

        max_value = gt_b_map_0.max()
        min_value = gt_b_map_0.min()

        gt_b_map_0 = (gt_b_map_0 - min_value) / max_value
        gt_heatmap_0 = np.uint8(255 * gt_b_map_0)
        gt_heatmap_0 = cv2.applyColorMap(gt_heatmap_0, cv2.COLORMAP_HOT)

        index_str1 = str(count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_map/' + index_str1 + '_gt_ext.png'
        cv2.imwrite(output_path_str1, gt_heatmap_0)

        max_value = gt_b_map_1.max()
        min_value = gt_b_map_1.min()

        gt_b_map_1 = (gt_b_map_1 - min_value) / max_value
        gt_heatmap_1 = np.uint8(255 * gt_b_map_1)
        gt_heatmap_1 = cv2.applyColorMap(gt_heatmap_1, cv2.COLORMAP_HOT)

        index_str2 = str(count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_map/' + index_str2 + '_gt_ext.png'
        cv2.imwrite(output_path_str2, gt_heatmap_1)
        #
        # ##################################              pred_b_ext          ####################################################
        pred_b_map = torch.sum(pred_b_ext.data, dim=1)
        pred_b_map = pred_b_map.data.cpu().numpy()
        pred_b_map_0 = pred_b_map[0, :, :]
        pred_b_map_1 = pred_b_map[1, :, :]

        max_value = pred_b_map_0.max()
        min_value = pred_b_map_0.min()

        pred_b_map_0 = (pred_b_map_0 - min_value) / max_value
        pred_heatmap_0 = np.uint8(255 * pred_b_map_0)
        pred_heatmap_0 = cv2.applyColorMap(pred_heatmap_0, cv2.COLORMAP_HOT)

        index_str1 = str(count).zfill(6)
        output_path_str1 = '/home/allen/mmseg_viz/boundary_map/' + index_str1 + '_pred_ext.png'
        cv2.imwrite(output_path_str1, pred_heatmap_0)

        max_value = pred_b_map_1.max()
        min_value = pred_b_map_1.min()

        pred_b_map_1 = (pred_b_map_1 - min_value) / max_value
        pred_heatmap_1 = np.uint8(255 * pred_b_map_1)
        pred_heatmap_1 = cv2.applyColorMap(pred_heatmap_1, cv2.COLORMAP_HOT)

        index_str2 = str(count + 1).zfill(6)
        output_path_str2 = '/home/allen/mmseg_viz/boundary_map/' + index_str2 + '_pred_ext.png'
        cv2.imwrite(output_path_str2, pred_heatmap_1)
        #################################              end          ####################################################

    # reshape
    gt_b = gt_b.view(n, c, -1)  # [1,10,50176]
    pred_b = pred_b.view(n, c, -1)
    gt_b_ext = gt_b_ext.view(n, c, -1)
    pred_b_ext = pred_b_ext.view(n, c, -1)

    pred_sum = torch.sum(pred_b, dim=2) + 0.0001
    gt_sum = torch.sum(gt_b, dim=2) + 0.0001
    # Precision, Recall
    P = torch.sum(pred_b * gt_b_ext, dim=2) / pred_sum + 0.0001
    R = torch.sum(pred_b_ext * gt_b, dim=2) / gt_sum + 0.0001

    # Boundary F1 Score
    BF1 = 2 * P * R / (P + R)
    loss = torch.mean(torch.sum(1 - BF1, dim=1), dim=0)

    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class MultiLoss(nn.Module):
    """BoundaryCrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 cross_entropy_loss_weight=1.0,
                 boundary_loss_weight=1.0,
                 theta0=3,
                 theta=5):
        super(MultiLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.cross_entropy_loss_weight = cross_entropy_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.class_weight = class_weight
        self.theta0 = theta0
        self.theta = theta
        self.count = 0
        self.seg_loss_sum = 0
        self.edge_loss_sum = 0

        if self.use_sigmoid:
            self.cross_entropy_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cross_entropy_criterion = mask_cross_entropy
        else:
            self.cross_entropy_criterion = cross_entropy

        self.boundary_criterion = boundary
        #self.boundary_criterion = test
    def forward(self,
                cls_score,
                cls_binary_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        cross_entropy_loss_cls = self.cross_entropy_loss_weight * self.cross_entropy_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        boundary_loss_cls = self.boundary_loss_weight * self.boundary_criterion(
            self.count,
            cls_binary_score,
            label,
            self.theta0,
            self.theta)
        self.count += 2
        self.edge_loss_sum = self.edge_loss_sum + boundary_loss_cls.data.cpu()
        self.seg_loss_sum = self.seg_loss_sum + cross_entropy_loss_cls.data.cpu()
        if self.count % 100 == 0:
            f = open('/home/allen/losses/loss.txt', 'a')
            f.write(str(self.count // 2)+ "\t" + str(self.seg_loss_sum.data.numpy() / 50)+"\t"+str(self.edge_loss_sum.data.numpy() / 50 / 0.15)+"\n")
            f.close()
            self.edge_loss_sum = 0
            self.seg_loss_sum = 0
        return cross_entropy_loss_cls + boundary_loss_cls