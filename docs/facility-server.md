# Facility GPU server (ibif-napari-server)

One GPU workstation in the facility runs Cellpose for every IBIF napari user on the
campus network. Users keep napari on their own computers. When they click
**Run Cellpose**, only the selected region is sent to the workstation; the labels come
back as a layer. Jobs from all users share one queue and run one at a time.

## 1. Hardware

- An NVIDIA GPU with 16 GB of memory or more (24 GB is better for large 3D regions).
- 64 GB of system RAM, an SSD, and a wired campus network connection.
- Windows 10/11 or Ubuntu Linux.
- A fixed address or host name on the campus network, from Creighton DoIT.

## 2. Install

```bash
conda create -n ibif-napari python=3.11 -y
conda activate ibif-napari
# CUDA build of PyTorch: pick the command for your system at https://pytorch.org/get-started/locally/
pip install "git+https://github.com/CU-IBIF/Napari_Cellpose"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ibif-napari --download-models          # fetch the Cellpose weights once
```

The Python check must print `True` and the GPU's name. If it prints `False`, PyTorch was
installed without CUDA support: reinstall it with the command from pytorch.org.

## 3. Add users

Each person gets their own access token. The server stores only a hash of each token,
so a lost token cannot be shown again; just issue a new one.

```bash
ibif-napari-server add-user alice      # prints alice's token once: send it privately
ibif-napari-server list-users
ibif-napari-server remove-user alice   # takes effect within a minute
```

Run these as the same account that runs the server: the users file is
`~/.ibif-napari-server/users.json` in that account's home folder. Use `--config-dir`
to keep it somewhere else.

## 4. Try it

```bash
ibif-napari-server run --allow 10.0.0.0/8
```

Use Creighton's campus and VPN address ranges for `--allow` (ask DoIT); the option can be
repeated. Connections from other addresses are refused. The log shows each job with the
user's name, never their token.

On a user's computer, in the napari Cellpose panel:

1. Click **Servers…**, enter a name (e.g. *Facility GPU*), the address
   (`http://<workstation>:8765`) and the user's token.
2. Click **Test connection**. It should report the GPU, the Cellpose version and the
   user's name.
3. Choose the server under **Run on** and run Cellpose as usual. **Cancel** stops the
   run on the server immediately.

Users do not need Cellpose or a GPU on their own computers to use the server.

## 5. Security

- **Network:** limit access with `--allow` and with the operating system firewall.
  Install scripts below open the port for the allowed networks only.
- **Encryption:** plain HTTP is acceptable only on a trusted network. For HTTPS, ask DoIT
  for a certificate for the workstation's host name and start the server with
  `--certfile cert.pem --keyfile key.pem`; users then enter `https://…`. With a
  self-signed certificate, users also pick the certificate file under **Certificate** in
  the Servers dialog.
- **Data:** uploads are held in memory only while a job runs. Results are deleted when the
  user fetches them, or after one hour. Nothing is written to disk.
- **Models:** users can choose built-in Cellpose models, or custom models the admin places
  in a folder given with `--models-dir`. Other file paths are refused.
- **Restricted data:** whether HIPAA-regulated or other restricted data may be processed
  on this workstation is a decision for Creighton DoIT and the compliance office.

## 6. Run at start-up

**Windows:** in an administrator PowerShell, from the `deploy` folder of this repository:

```powershell
.\install-windows-task.ps1 -EnvPath C:\Users\ibif\miniconda3\envs\ibif-napari -Allow 10.0.0.0/8
```

This starts the server at boot under the Windows account you enter, restarts it if it
stops, and opens the firewall port for the allowed networks. The log is
`C:\ProgramData\ibif-napari-server.log`. Remove it with `-Uninstall`.

**Linux:** edit `deploy/ibif-napari-server.service` (user, environment path, `--allow`),
then:

```bash
sudo cp deploy/ibif-napari-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ibif-napari-server
journalctl -u ibif-napari-server -f
```

## 7. Update

```bash
conda activate ibif-napari
pip install --upgrade "git+https://github.com/CU-IBIF/Napari_Cellpose"
```

Then restart the scheduled task or the systemd service. Ask users to update too, so
both sides use the same Cellpose version.

## Server options

| Option | Meaning |
| --- | --- |
| `--allow CIDR` | Only accept these networks (repeatable) |
| `--port N` | Port to listen on (default 8765) |
| `--certfile`, `--keyfile` | Serve HTTPS with this certificate and key |
| `--device auto/gpu/cpu` | Use the GPU if present (default), always, or never |
| `--preload MODEL` | Model to load at start (default: newest built-in) |
| `--models-dir DIR` | Folder of custom Cellpose models users may choose by file name |
| `--max-mb N` | Largest upload in MB (default 2048) |
| `--log-file FILE` | Also write the log to this file |

## Troubleshooting

- **"cannot reach …"** on the user's side: check the address and port, that the server is
  running, and that the user's network is within `--allow` and the firewall rule.
- **"access token was not accepted":** the token was mistyped or the user was removed.
  Issue a new one with `add-user`.
- **Status says CPU instead of GPU:** PyTorch cannot see the GPU; see section 2.
- **"stopped unexpectedly … out of memory":** the region was too large for the GPU. Use a
  smaller region, or the stitched 2D mode.
