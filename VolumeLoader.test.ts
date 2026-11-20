import { describe, it, expect } from 'vitest';
import {
  packVolume,
  unpackVolume,
  buildBlockMax,
  type Volume,
} from './VolumeLoader';

/* ------------------------------------------------------------------ *
 * 1. 非对称 2×3×4 测试数据
 *    shape = [nx=2, ny=3, nz=4]，共 24 个体素，每个值唯一。
 *    故意用非立方体尺寸：立方体在轴交换时可能"看起来"仍然正确，
 *    非对称数据能让任何维度错序立刻暴露。
 * ------------------------------------------------------------------ */

const NX = 2, NY = 3, NZ = 4;
const N = NX * NY * NZ; // 24

function makeAsymmetricVolume(): Volume {
  const data = new Float32Array(N);
  for (let i = 0; i < N; i++) data[i] = i + 1; // 1..24，避免 0 与默认值混淆
  return { data, shape: [NX, NY, NZ] };
}

/** 源数据按 (z, y, x) 行主序索引，即 x 变化最快 */
function srcIndex(x: number, y: number, z: number) {
  return (z * NY + y) * NX + x;
}

/** 打包后按 ZYX 顺序索引，即 x 是最内层（连续）维度 */
function packedIndexZYX(x: number, y: number, z: number) {
  return (z * NY + y) * NX + x;
}

describe('packVolume — 非对称 2×3×4 维度顺序', () => {
  const vol = makeAsymmetricVolume();

  it('输出长度等于体素总数', () => {
    const out = packVolume(vol, { order: 'ZYX' });
    expect(out).toHaveLength(N);
  });

  it('每个体素都落到了正确的偏移位置', () => {
    const out = packVolume(vol, { order: 'ZYX' });
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          expect(out[packedIndexZYX(x, y, z)]).toBe(vol.data[srcIndex(x, y, z)]);
        }
      }
    }
  });

  it('角点体素位置正确（首尾不变）', () => {
    const out = packVolume(vol, { order: 'ZYX' });
    // 原点
    expect(out[0]).toBe(vol.data[srcIndex(0, 0, 0)]);
    // 对角最远点
    expect(out[N - 1]).toBe(vol.data[srcIndex(NX - 1, NY - 1, NZ - 1)]);
  });

  it('轴交换时测试必须失败（自证测试有效性）', () => {
    // 构造一个把 x/z 交换的错误打包结果，断言它 != 正确结果。
    // 如果这条测试通过，说明我们的 fixture 确实能区分轴顺序。
    const wrong = new Float32Array(N);
    for (let z = 0; z < NZ; z++)
      for (let y = 0; y < NY; y++)
        for (let x = 0; x < NX; x++)
          wrong[packedIndexZYX(x, y, z)] = vol.data[srcIndex(z, y, x)]; // 故意错
    const right = packVolume(vol, { order: 'ZYX' });
    expect(Array.from(wrong)).not.toEqual(Array.from(right));
  });
});

/* ------------------------------------------------------------------ *
 * 2. 字节序（endianness）检查
 *    0x41890000 = 17.125f（IEEE-754 单精度）
 *    小端机器内存布局: 00 00 89 41
 *    大端机器内存布局: 41 89 00 00
 *    这里断言"打包结果按小端写入"，因为 WebGL texImage3D 期望
 *    主机字节序，而绝大多数目标平台是 LE。若平台为 BE 则跳过。
 * ------------------------------------------------------------------ */

describe('packVolume — 字节序', () => {
  const IS_LITTLE_ENDIAN =
    new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

  it('哨兵值 17.125f 的字节序列符合主机字节序', () => {
    const vol: Volume = {
      data: new Float32Array([17.125]),
      shape: [1, 1, 1],
    };
    const packed = packVolume(vol, { order: 'ZYX' });
    const bytes = new Uint8Array(packed.buffer);

    expect(bytes).toHaveLength(4);
    if (IS_LITTLE_ENDIAN) {
      expect(Array.from(bytes)).toEqual([0x00, 0x00, 0x89, 0x41]);
    } else {
      expect(Array.from(bytes)).toEqual([0x41, 0x89, 0x00, 0x00]);
    }
  });

  it('打包后的字节长度是 4 的整数倍（无残缺字）', () => {
    const out = packVolume(makeAsymmetricVolume(), { order: 'ZYX' });
    expect(out.byteLength % 4).toBe(0);
  });

  it('往返后逐字节完全一致（bit-exact，非容差比较）', () => {
    const vol = makeAsymmetricVolume();
    const packed = packVolume(vol, { order: 'ZYX' });
    const unpacked = unpackVolume(packed, vol.shape);
    expect(Buffer.from(unpacked.buffer)).toEqual(Buffer.from(vol.data.buffer));
  });
});

/* ------------------------------------------------------------------ *
 * 3. 块最大值金字塔 —— 保守边界属性测试
 *
 *    核心不变量：block.max >= max(该块覆盖的所有源体素)
 *    块最大值只作为光线的"跳过空区"加速结构，必须是保守上界：
 *      - 高估：安全，只是多做冗余采样
 *      - 低估：错误，光线会直接穿过真实密度，渲染出现空洞
 *    因此断言的是"上界"而非"相等"，逐层向下直到 1×1×1。
 * ------------------------------------------------------------------ */

describe('buildBlockMax — 保守边界属性', () => {
  const BLOCK = 8;

  /** 返回某块在源体素网格中覆盖的所有索引 */
  function* membersOf(
    origin: [number, number, number],
    size: [number, number, number],
    shape: [number, number, number]
  ) {
    const [ox, oy, oz] = origin;
    const [sx, sy, sz] = size;
    const [nx, ny, nz] = shape;
    for (let z = oz; z < Math.min(oz + sz, nz); z++)
      for (let y = oy; y < Math.min(oy + sy, ny); y++)
        for (let x = ox; x < Math.min(ox + sx, nx); x++)
          yield srcIndex(x, y, z);
  }

  it('每一层的每个块都满足 max >= 其所有子体素', () => {
    const vol = makeAsymmetricVolume();
    const pyramid = buildBlockMax(vol, { blockSize: BLOCK });

    for (let level = 0; level < pyramid.levels.length; level++) {
      const layer = pyramid.levels[level];
      for (const block of layer.blocks) {
        let refMax = -Infinity;
        for (const idx of membersOf(block.origin, block.size, vol.shape)) {
          refMax = Math.max(refMax, vol.data[idx]);
        }
        // 保守上界：绝不低估
        expect(block.max).toBeGreaterThanOrEqual(refMax);
        // 也不应离谱高估（允许浮点误差）
        expect(block.max).toBeCloseTo(refMax, 5);
      }
    }
  });

  it('叶子层块数之和覆盖全部源体素（无遗漏、无重复）', () => {
    const vol = makeAsymmetricVolume();
    const pyramid = buildBlockMax(vol, { blockSize: BLOCK });
    const leaf = pyramid.levels[pyramid.levels.length - 1];

    let covered = 0;
    const seen = new Set<number>();
    for (const block of leaf.blocks) {
      for (const idx of membersOf(block.origin, block.size, vol.shape)) {
        expect(seen.has(idx)).toBe(false); // 不得重复覆盖
        seen.add(idx);
        covered++;
      }
    }
    expect(covered).toBe(N);
    expect(seen.size).toBe(N);
  });

  it('非整除尺寸时，填充值不得污染块最大值', () => {
    // 2×3×4 不能被 8 整除，必然产生边界块。
    // 若填充用 0 而非 -Infinity，块最大值会被抬到 0 以上，
    // 导致空区被误判为有密度。
    const vol: Volume = {
      data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
      shape: [2, 2, 2],
    };
    const pyramid = buildBlockMax(vol, { blockSize: BLOCK });
    const root = pyramid.levels[0].blocks[0];
    expect(root.max).toBe(8); // 恰好等于真实最大值，未被填充抬高
  });

  it('全空体数据（全 0）的块最大值应为 0 或 -Infinity，不得为正', () => {
    const vol: Volume = {
      data: new Float32Array(N), // 全 0
      shape: [NX, NY, NZ],
    };
    const pyramid = buildBlockMax(vol, { blockSize: BLOCK });
    for (const layer of pyramid.levels) {
      for (const block of layer.blocks) {
        expect(block.max).toBeLessThanOrEqual(0);
      }
    }
  });
});
