import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj, class_count):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.class_count = class_count
        self.mean_square = nn.MSELoss(reduction="sum")

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def get_class_prediction_loss(self, classes_pred, classes_target):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, self.class_count)   #(batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, self.class_count)   #(batch_size, S, S, 20)

        Returns:
        class_loss : scalar
        """

        ##### CODE #####

        return self.mean_square(
            torch.flatten(classes_pred, start_dim=-2),
            torch.flatten(classes_target, start_dim=-2)
        )

    def get_regression_loss(self, box_pred_response: torch.tensor, box_target_response: torch.tensor):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """

        ##### CODE #####

        xy_pred_response = box_pred_response[..., 0:2]
        xy_target_response = box_target_response[..., 0:2]

        wh_pred_response = box_pred_response[..., 2:4]
        wh_target_response = box_target_response[..., 2:4]

        # sqrt the width and height, first abs incase neg and add tiny val to prevent derivitive error, then give the sign back
        wh_pred_response = torch.sign(wh_pred_response) * torch.sqrt(torch.abs(wh_pred_response + 0.00001))
        wh_target_response = torch.sign(wh_target_response) * torch.sqrt(torch.abs(wh_target_response + 0.00001))

        # preform mse on xy and wh
        # box_loss = self.mean_square(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))
        xy_loss = self.mean_square(torch.flatten(xy_pred_response), torch.flatten(xy_target_response))
        wh_loss = self.mean_square(torch.flatten(wh_pred_response), torch.flatten(wh_target_response))

        return xy_loss + wh_loss

    def get_contain_conf_loss(self, box_pred_response: torch.tensor, box_target_response_iou: torch.tensor):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        contain_loss : scalar

        """

        ##### CODE #####

        # box_pred_response = torch.reshape(box_pred_response, (-1, 5))
        # box_target_response_iou = torch.reshape(box_target_response_iou, (-1, 5))

        return self.mean_square(torch.flatten(box_pred_response[..., 4:5]),
                                torch.flatten(box_target_response_iou[..., 4:5]))

    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, Bx5+class_count=?30)
        pred_tensor : (tensor) size (batch_size, S , S, Bx5+class_count=?30)
        no_object_mask : (tensor) size (batch_size, S , S, 1) # (tensor) size (batch_size, S , S, Bx5+class_count=?30)

        Returns:
        no_object_loss : scalar

        Hints:
        1) Create a 2 tensors no_object_prediction and no_object_target which only have the
        values which have no object.
        2) Have another tensor no_object_prediction_mask of the same size such that
        mask with respect to both confidences of bounding boxes set to 1.
        3) Create 2 tensors which are extracted from no_object_prediction and no_object_target using
        the mask created above to find the loss.
        """

        ##### CODE #####

        no_object_prediction_box_1 = no_object_mask * pred_tensor[..., 4:5]
        no_object_prediction_box_2 = no_object_mask * pred_tensor[..., 9:10]

        no_object_target = no_object_mask * target_tensor[..., 4:5]

        no_object_loss_box_1 = self.mean_square(
            torch.flatten(no_object_prediction_box_1),
            torch.flatten(no_object_target)
        )

        no_object_loss_box_2 = self.mean_square(
            torch.flatten(no_object_prediction_box_2),
            torch.flatten(no_object_target)
        )

        return no_object_loss_box_1 + no_object_loss_box_2

    def find_best_iou_boxes(self, box_target, box_pred):

        """
        Parameters:
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) Set the corresponding contains_object_response_mask of the bounding box with the max iou
        of the 2 bounding boxes of each grid cell to 1.
        3) For finding iou's use the compute_iou function
        4) Before using compute preprocess the bounding box coordinates in such a way that
        if for a Box b the coordinates are represented by [x, y, w, h] then
        x, y = x/S - 0.5*w, y/S - 0.5*h ; w, h = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        5) Set the confidence of the box_target_iou of the bounding box to the maximum iou

        """

        ##### CODE #####
        contains_object_response_mask = torch.ones(size=[int(box_pred.size()[0] / 2)], device="cuda:0")
        box_target_iou = torch.zeros_like(contains_object_response_mask)

        box_pred_corners = torch.zeros_like(box_pred)

        x = box_pred[..., 0:1]
        y = box_pred[..., 1:2]
        w = box_pred[..., 2:3] / 2
        h = box_pred[..., 3:4] / 2

        box_pred_corners[..., 0:1] = x - w
        box_pred_corners[..., 1:2] = y - h
        box_pred_corners[..., 2:3] = x + w
        box_pred_corners[..., 3:4] = y + h

        box_target_corners = torch.zeros_like(box_target)

        x = box_target[..., 0:1]
        y = box_target[..., 1:2]
        w = box_target[..., 2:3] / 2
        h = box_target[..., 3:4] / 2

        box_target_corners[..., 0:1] = x - w
        box_target_corners[..., 1:2] = y - h
        box_target_corners[..., 2:3] = x + w
        box_target_corners[..., 3:4] = y + h

        intersection_x1 = torch.max(box_pred_corners[..., 0:1], box_target_corners[..., 0:1])
        intersection_y1 = torch.max(box_pred_corners[..., 1:2], box_target_corners[..., 1:2])
        intersection_x2 = torch.min(box_pred_corners[..., 2:3], box_target_corners[..., 2:3])
        intersection_y2 = torch.min(box_pred_corners[..., 3:4], box_target_corners[..., 3:4])

        intersection_area = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)
        pred_area = abs((box_pred_corners[..., 2:3] - box_pred_corners[..., 0:1]) * (
                    box_pred_corners[..., 3:4] - box_pred_corners[..., 1:2]))
        target_area = abs((box_target_corners[..., 2:3] - box_target_corners[..., 0:1]) * (
                    box_target_corners[..., 3:4] - box_target_corners[..., 1:2]))

        # make sure denominator is not zero
        ious = intersection_area / (target_area + pred_area - intersection_area + 0.000001)

        for i in range(int(ious.size()[0] / 2)):
            idx = i * 2
            if ious[idx][0] > ious[idx + 1][0]:
                contains_object_response_mask[..., i + 1:i + 2] -= contains_object_response_mask[..., i + 1:i + 2]
                box_target_iou[..., i:i + 1] = ious[idx][0]
            else:
                # contains_object_response_mask[..., i:i + 1] -= contains_object_response_mask[..., i:i + 1]
                box_target_iou[..., i:i + 1] = ious[idx + 1][0]

        return box_target_iou, contains_object_response_mask

    def forward(self, pred_tensor: torch.tensor, target_tensor: torch.tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+class_count=?30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            self.class_count - number of classes

        target_tensor: (tensor) size(batchsize,S,S,Bx5+class_count=?30)

        Returns:
        Total Loss
        '''

        N = pred_tensor.size()[0]

        total_loss = None

        # Create 2 tensors contains_object_mask and no_object_mask
        # of size (Batch_size, S, S) such that each value corresponds to if the confidence of having
        # an object > 0 in the target tensor.

        ##### CODE #####

        # (N, S ,S, 1)
        contains_object_mask = target_tensor[..., 4].unsqueeze(3)

        # Create a tensor contains_object_pred that corresponds to
        # to all the predictions which seem to confidence > 0 for having an object
        # Split this tensor into 2 tensors :
        # 1) bounding_box_pred : Contains all the Bounding box predictions of all grid cells of all images
        # 2) classes_pred : Contains all the class predictions for each grid cell of each image
        # Hint : Use contains_object_mask

        ##### CODE #####

        # contains_object_pred1 = contains_object_mask * pred_tensor[..., 0:5]
        # contains_object_pred2 = contains_object_mask * pred_tensor[..., 5:10]
        bounding_box_pred1 = torch.ravel(contains_object_mask * pred_tensor[..., 0:5])
        bounding_box_pred2 = torch.ravel(contains_object_mask * pred_tensor[..., 5:10])
        bounding_box_pred = torch.reshape(torch.hstack([bounding_box_pred1, bounding_box_pred2]), (-1, 5))

        classes_pred = contains_object_mask * pred_tensor[..., self.B * 5:]

        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.

        ##### CODE #####

        bounding_box_target1 = torch.ravel(contains_object_mask * target_tensor[..., 0:5])
        bounding_box_target2 = torch.ravel(contains_object_mask * target_tensor[..., 5:10])
        bounding_box_target = torch.reshape(torch.hstack([bounding_box_target1, bounding_box_target2]), (-1, 5))

        classes_target = contains_object_mask * target_tensor[..., self.B * 5:]

        # Compute the No object loss here

        ##### CODE #####

        no_object_loss = self.get_no_object_loss(target_tensor, pred_tensor, 1 - contains_object_mask)

        # Compute the iou's of all bounding boxes and the mask for which bounding box
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.

        ##### CODE #####

        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(bounding_box_target[..., 0:4],
                                                                                 bounding_box_pred[..., 0:4])

        corm_origional_shape = torch.reshape(contains_object_response_mask, [N, self.S, self.S, 1])

        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        # Hint : Use contains_object_response_mask

        ##### CODE #####

        # mask out only the predictions where there is an obj, and also pick only the box with higher iou
        # (N, S ,S, [x,y,w,h])
        box_prediction_response = contains_object_mask * (
                (1 - corm_origional_shape) * pred_tensor[..., 0:5]
                + corm_origional_shape * pred_tensor[..., 5:10]
        )

        # mask out only the predictions where there is an obj, and also pick only the box with higher iou
        # (N, S ,S, [x,y,w,h])
        box_target_response = contains_object_mask * (
                (1 - corm_origional_shape) * target_tensor[..., 0:5]
                + corm_origional_shape * target_tensor[..., 5:10]
        )

        # Find the class_loss, containing object loss and regression loss

        ##### CODE #####

        # get class loss
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)

        # get reg loss
        reg_loss = self.get_regression_loss(box_prediction_response, box_target_response)

        # get obj existing loss
        object_loss = self.get_contain_conf_loss(box_prediction_response, box_target_response)

        total_loss = (
                self.l_coord * reg_loss +
                object_loss +
                self.l_noobj * no_object_loss +
                class_loss
        )

        return total_loss
