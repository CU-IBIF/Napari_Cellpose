# GPU clusters (Slurm, e.g. NSF ACCESS)

IBIF napari can start a GPU job on a Slurm cluster and run Cellpose there, while you keep
viewing and drawing regions in napari on your own computer. You click **Connect**, the job
waits in the cluster's queue, and once it runs, **Run Cellpose** works exactly as it does
locally. Only the selected region is sent. **Disconnect** ends the job.

Everything goes over one SSH login: your key, then two-factor if the cluster asks. That
one connection submits the job, follows it in the queue, and carries a tunnel from your
computer to the job's compute node. Nothing else needs installing on your computer; this
works the same on Windows, macOS and Linux.

## 1. Get an account on a GPU cluster

For Creighton users the free route is an **NSF ACCESS** allocation:

1. The PI requests an *Explore* allocation at allocations.access-ci.org (a short
   description and CVs) and exchanges the credits for time on a system with GPUs.
2. The PI adds users to the allocation. Each user then gets an account on that system.
3. Each user sets up SSH login as that system's user guide describes. Many systems want
   your **public SSH key** uploaded through their web portal, and some also require
   two-factor authentication.

To create an SSH key (Windows PowerShell, macOS or Linux terminal):

```bash
ssh-keygen -t ed25519
```

Upload the `.pub` file it creates (e.g. `~/.ssh/id_ed25519.pub`), never the private key.

## 2. Add the cluster in napari

In the Cellpose panel click **Clusters…** and fill in:

| Field | What to enter |
| --- | --- |
| Name | Any name, e.g. *Anvil GPU* |
| Login host | The system's SSH login address, from its user guide |
| Username | Your username on that system (often different from your Creighton one) |
| SSH key file | Leave empty to use your SSH agent or `~/.ssh/id_*` keys |
| Setup lines | Commands needed first, e.g. `module load python/3.11` if the default Python is older than 3.10 |
| Work folder | Where the server and its logs live on the cluster (default `~/.ibif-napari-cluster`) |
| Activate environment | Leave empty to use the environment **Install** creates |
| Partition | The GPU partition (queue) name, from the system's user guide |
| Account (allocation) | Your allocation's account name on that system |
| GPUs (`--gpus=`) | Usually `1`; some systems want a type, e.g. `a100:1`, as their guide shows |
| Time limit | Longest a session can last, e.g. `04:00:00`. The job ends earlier when unused |
| CPUs, Memory | Defaults (8 CPUs, 32G) suit Cellpose |
| Extra #SBATCH lines | Anything else the system requires, e.g. `#SBATCH --qos=...` |
| Stop after idle (min) | The job ends after this many minutes without use (default 15) |
| Extra server options | Rarely needed, e.g. `--models-dir /shared/cellpose-models` |

Then:

1. **Test login** logs in and lists the system's partitions and GPU types. It also says
   whether IBIF napari is installed there yet.
2. **Install / update on cluster** creates a Python environment in the work folder,
   installs the Cellpose server (without napari), and downloads the Cellpose weights. It
   runs on the login node, because compute nodes often have no internet. The log shows
   each step; the last lines report the PyTorch version and its CUDA build.
3. **Job script** shows exactly what will be submitted.

## 3. Use it

1. Under **Run on**, choose *<name> (cluster)* and click **Connect**.
2. Answer the two-factor prompt if the cluster asks. The first time, napari shows the
   cluster's host-key fingerprint; compare it with the one the system documents.
3. The status line shows the job waiting in the queue, then starting the server, then
   *Connected … job 12345*.
4. Run Cellpose as usual. Regions, Z planes, every time point and Cancel all work.
5. Click **Disconnect** when you are done. The job also ends by itself after the idle
   time, at its time limit, or when napari closes.

Allocation credits are charged while the job runs, including the idle minutes, so
disconnect when you are finished and keep the idle time short.

## Security

- Everything travels over your SSH login. Nothing is opened to the internet.
- Each session's server accepts only a one-off token, stored in your work folder with
  permissions only you can read.
- Images are held in memory only while they are segmented; results are deleted once
  fetched.
- National clusters are generally not approved for HIPAA-regulated or other restricted
  data. Use the facility workstation for such projects, subject to Creighton's policies.

## Command line

The same steps work from a terminal, which is useful for troubleshooting:

```bash
ibif-napari-cluster add "Anvil GPU" --host <login host> --user <username> \
    --partition <gpu partition> --account <allocation account>
ibif-napari-cluster test "Anvil GPU"
ibif-napari-cluster install "Anvil GPU"
ibif-napari-cluster script "Anvil GPU"     # print the job script
ibif-napari-cluster connect "Anvil GPU"    # start a job and keep a tunnel open until Ctrl+C
```

## Troubleshooting

- **"login … failed: no accepted key":** the public key is not on the cluster yet, or the
  profile points at the wrong key file.
- **"the host key … has CHANGED":** do not continue. Check with the cluster's support; if
  the change is expected, remove its line from `~/.ibif-napari/known_hosts`.
- **"refused the SSH tunnel":** the system does not allow SSH port forwarding to compute
  nodes. Ask its support; not every system permits it.
- **"job … ended before starting the server":** the message includes the end of the job
  log (`job-<id>.log` in the work folder). Typical causes are a wrong partition or
  account name, or a missing Setup line (module load).
- **Status says CPU instead of GPU:** the job did not get a GPU (check the partition and
  GPUs fields), or PyTorch was installed without CUDA. The Install log shows the CUDA
  build; some systems need `module load cuda` in Setup lines.
- **Long queue waits:** try a shorter time limit or another GPU partition, if the system
  has one.
