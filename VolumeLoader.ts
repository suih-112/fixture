export interface Volume {
  /** 源数据，按 (z, y, x) 行主序，x 变化最快 */
  data: Float32Array;
  /** [nx, ny, nz] */
  shape: [number, number, number];
}

export type AxisOrder = 'ZYX' | 'XYZ';

export function packVolume(
  vol: Volume,
  opts: { order: AxisOrder }
): Float32Array {
  const [nx, ny, nz] = vol.shape;
  const out = new Float32Array(nx * ny * nz);

  if (opts.order === 'ZYX') {
    // 目标布局：x 最内层连续。源数据本就是 x 最快，
    // 因此这是恒等拷贝；若源为 (time, level, lat, lon) 则在此转置。
    out.set(vol.data);
  } else {
    // XYZ：z 最内层
    for (let z = 0; z < nz; z++)
      for (let y = 0; y < ny; y++)
        for (let x = 0; x < nx; x++)
          out[(x * ny + y) * nz + z] = vol.data[(z * ny + y) * nx + x];
  }
  return out;
}

export function unpackVolume(
  packed: Float32Array,
  shape: [number, number, number]
): Float32Array {
  const [nx, ny, nz] = shape;
  const out = new Float32Array(nx * ny * nz);
  out.set(packed);
  return out;
}

export interface Block {
  origin: [number, number, number];
  size: [number, number, number];
  max: number;
}

export interface BlockMaxPyramid {
  levels: { blockSize: number; blocks: Block[] }[];
}

export function buildBlockMax(
  vol: Volume,
  opts: { blockSize: number }
): BlockMaxPyramid {
  const [nx, ny, nz] = vol.shape;
  const levels: BlockMaxPyramid['levels'] = [];

  let blockSize = opts.blockSize;
  // 顶层：整卷一个块
  levels.push({
    blockSize,
    blocks: [computeBlock(vol, [0, 0, 0], [nx, ny, nz])],
  });

  // 逐层细分，直到块尺寸 <= 1
  while (blockSize > 1) {
    const childSize = Math.max(1, Math.floor(blockSize / 2));
    const blocks: Block[] = [];
    for (let z = 0; z < nz; z += childSize)
      for (let y = 0; y < ny; y += childSize)
        for (let x = 0; x < nx; x += childSize)
          blocks.push(
            computeBlock(
              vol,
              [x, y, z],
              [
                Math.min(childSize, nx - x),
                Math.min(childSize, ny - y),
                Math.min(childSize, nz - z),
              ]
            )
          );
    levels.push({ blockSize: childSize, blocks });
    blockSize = childSize;
  }

  return { levels };
}

function computeBlock(
  vol: Volume,
  origin: [number, number, number],
  size: [number, number, number]
): Block {
  const [nx, ny, nz] = vol.shape;
  const [ox, oy, oz] = origin;
  const [sx, sy, sz] = size;
  let max = -Infinity; // 关键：用 -Infinity 而非 0，避免填充污染
  for (let z = oz; z < Math.min(oz + sz, nz); z++)
    for (let y = oy; z < nz && y < Math.min(oy + sy, ny); y++)
      for (let x = ox; x < Math.min(ox + sx, nx); x++)
        max = Math.max(max, vol.data[(z * ny + y) * nx + x]);
  return { origin, size, max };
}
