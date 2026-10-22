"""Write minimal Leica LIF files for tests (just enough for liffile / bioio-lif)."""
import struct

import numpy as np


def _dimension(dim_id, n, length_m, bytes_inc):
    return (
        f'<DimensionDescription DimID="{dim_id}" NumberOfElements="{n}" Origin="0" '
        f'Length="{length_m}" Unit="m" BitInc="0" BytesInc="{bytes_inc}"/>'
    )


def _image_xml(name, data, channels, xy_um, z_um, t_s, block_id):
    """data: uint16 array (T, Z, C, Y, X)."""
    nt, nz, nc, ny, nx = data.shape
    plane = ny * nx * 2
    chans = "".join(
        f'<ChannelDescription DataType="0" ChannelTag="0" Resolution="16" '
        f'NameOfMeasuredQuantity="" Min="0" Max="65535" Unit="" LUTName="{lut}" '
        f'IsLUTInverted="0" BytesInc="{c * plane}" BitInc="0"/>'
        for c, lut in enumerate(channels)
    )
    dims = _dimension(1, nx, (nx - 1) * xy_um * 1e-6, 2)
    dims += _dimension(2, ny, (ny - 1) * xy_um * 1e-6, nx * 2)
    if nz > 1:
        dims += _dimension(3, nz, (nz - 1) * z_um * 1e-6, plane * nc)
    if nt > 1:
        dims += _dimension(4, nt, (nt - 1) * t_s, plane * nc * nz)
    return (
        f'<Element Name="{name}"><Data><Image><ImageDescription>'
        f"<Channels>{chans}</Channels><Dimensions>{dims}</Dimensions>"
        f'</ImageDescription></Image></Data>'
        f'<Memory Size="{data.nbytes}" MemoryBlockID="{block_id}"/><Children/></Element>'
    )


def write_lif(path, images):
    """images: list of (name, uint16 array (T, Z, C, Y, X), channel LUT names, xy_um, z_um, t_s)."""
    blocks = [("MemBlock_0", b"")]
    children = ""
    for i, (name, data, channels, xy_um, z_um, t_s) in enumerate(images, start=1):
        block_id = f"MemBlock_{i}"
        children += _image_xml(name, data, channels, xy_um, z_um, t_s, block_id)
        # X fastest, then Y, then channel, then Z, then T: the order of (T, Z, C, Y, X).
        blocks.append((block_id, np.ascontiguousarray(data, "<u2").tobytes()))
    xml = (
        '<LMSDataContainerHeader Version="2"><Element Name="project"><Data><Experiment/></Data>'
        '<Memory Size="0" MemoryBlockID="MemBlock_0"/>'
        f"<Children>{children}</Children></Element></LMSDataContainerHeader>"
    )
    with open(path, "wb") as f:
        text = xml.encode("utf-16-le")
        f.write(struct.pack("<IIBI", 0x70, len(text) + 5, 0x2A, len(xml)))
        f.write(text)
        for block_id, payload in blocks:
            bid = block_id.encode("utf-16-le")
            f.write(struct.pack("<IIBQBI", 0x70, 0, 0x2A, len(payload), 0x2A, len(block_id)))
            f.write(bid)
            f.write(payload)
