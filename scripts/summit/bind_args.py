import sys

ranks = int(sys.argv[1])
cpu_bind = "/".join(
    ["0-27", "28-55", "56-83", "88-115", "116-143", "144-171"][:ranks]
)
mem_bind = "/".join(["0", "0", "0", "8", "8", "8"][:ranks])
gpu_bind = "/".join(["0", "1", "2", "3", "4", "5"][:ranks])
nic_bind = "/".join(
    [
        "mlx5_0,mlx5_1",
        "mlx5_0,mlx5_1",
        "mlx5_0,mlx5_1",
        "mlx5_2,mlx5_3",
        "mlx5_2,mlx5_3",
        "mlx5_2,mlx5_3",
    ][:ranks]
)

print(
    "--cpu-bind {} --mem-bind {} --gpu-bind {} --nic-bind {}".format(
        cpu_bind, mem_bind, gpu_bind, nic_bind
    )
)
