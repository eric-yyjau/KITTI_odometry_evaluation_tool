## euler angle to 3x4 matrix
## or quaternion to 3x4 matrix
## preprocess before kitti evaluation.

from tools.pose_evaluation_utils import quat_pose_to_mat
import argparse
import numpy as np

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def euler2mat(vec):
    # print(f"vec: {vec}")
    rot = eulerAnglesToRotationMatrix(vec[:3])
    # print(f"rot: {rot}, trans: {vec[3:]}")
    mat = np.concatenate((rot, vec[3:].reshape(3,1)), axis=1)
    return mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change format from quaternion to kitti format')

    parser.add_argument('traj_file',     type=str,  help='the trajectory file')
    parser.add_argument('out_file',     type=str,  help='the output file name')
    # parser.add_argument('--result_dir', type=str, default='./data/',              help='Directory path of storing the odometry results')
    parser.add_argument('--action',  type=str, default=None, help='[ euler2mat | ]') 
    # parser.add_argument('--toCameraCoord',   type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to convert the pose to camera coordinate')
    parser.add_argument('--removeTime', type=bool, action="store_true", help='remove first column')

    args = parser.parse_args()
    poses_quat = np.genfromtxt(args.traj_file, delimiter=",")
    # poses_mat = [quat_pose_to_mat(v)[:3,:] for v in poses_quat] 
    if args.removeTime:
        poses_quat = poses_quat[1:, :]

    if args.action == 'euler2mat':
        poses_mat = [euler2mat(v)[:3,:] for v in poses_quat] 
        poses_mat = np.array(poses_mat)
        poses_mat = poses_mat.reshape(-1, 12)
    np.savetxt(args.out_file, poses_mat)

    # pose_eval = kittiOdomEval(args)
    # pose_eval.eval(toCameraCoord=args.toCameraCoord)   # set the value according to the predicted results