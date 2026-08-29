import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { useRef } from "react";
import type { Mesh } from "three";
import { useStore, activeSeries } from "../store";
import { COLORS, LABELS, type SeriesKey } from "../types";

/** Fixed overlay rather than 3D text anchored to each ball: labels projected
 *  from world space drift off-screen as soon as the user orbits the camera. */
function Legend() {
  const visible = useStore((s) => s.visible);
  return (
    <div className="legend">
      {activeSeries(visible).map((k) => (
        <span key={k}>
          <i style={{ background: COLORS[k] }} />
          {LABELS[k]}
        </span>
      ))}
    </div>
  );
}

const RADIUS = 0.11;
const SPACING = 0.85;

function Ball({ which, x }: { which: SeriesKey; x: number }) {
  const ref = useRef<Mesh>(null);

  // Read the height straight from the store each frame rather than putting the
  // playhead in React state: this runs at 60fps and must not re-render the tree.
  useFrame(() => {
    const { trajectory, frame, mode } = useStore.getState();
    if (!ref.current || !trajectory) return;
    const source = mode === "naive" && which !== "truth" ? trajectory.naive : trajectory.contact;
    const q = source[which]?.q;
    if (!q) return;

    const i = Math.floor(frame);
    const f = frame - i;
    const a = q[Math.min(i, q.length - 1)];
    const b = q[Math.min(i + 1, q.length - 1)];
    ref.current.position.y = a + (b - a) * f + RADIUS;
  });

  return (
    <group position={[x, 0, 0]}>
      <mesh ref={ref} castShadow>
        <sphereGeometry args={[RADIUS, 32, 32]} />
        <meshStandardMaterial
          color={COLORS[which]}
          roughness={0.35}
          metalness={0.1}
          emissive={COLORS[which]}
          emissiveIntensity={0.18}
        />
      </mesh>
    </group>
  );
}

function Rig() {
  const visible = useStore((s) => s.visible);
  const keys = activeSeries(visible);
  const offset = ((keys.length - 1) * SPACING) / 2;

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[4, 6, 4]} intensity={1.5} castShadow />
      <pointLight position={[-4, 3, -2]} intensity={0.4} />

      {keys.map((k, i) => (
        <Ball key={k} which={k} x={i * SPACING - offset} />
      ))}

      {/* The floor has to read as a solid surface, otherwise the naive rollout
          passing through it does not look like anything is wrong. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]} receiveShadow>
        <planeGeometry args={[9, 6]} />
        <meshStandardMaterial color="#161d26" roughness={0.9} metalness={0} />
      </mesh>

      <Grid
        args={[14, 14]}
        cellSize={0.4}
        cellThickness={0.6}
        cellColor="#2a3038"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#39424e"
        fadeDistance={16}
        infiniteGrid
        position={[0, 0, 0]}
      />
      <OrbitControls
        enablePan={false}
        minDistance={2}
        maxDistance={12}
        maxPolarAngle={Math.PI / 2.05}
        target={[0, 0.95, 0]}
      />
    </>
  );
}

/** Drives the shared clock. Lives inside the Canvas to reuse its rAF loop. */
function Clock() {
  useFrame((_, delta) => {
    const { playing, advance } = useStore.getState();
    if (playing) advance(Math.min(delta, 0.05));
  });
  return null;
}

export default function Scene() {
  return (
    <>
      <Canvas shadows camera={{ position: [0, 1.5, 4.6], fov: 44 }} dpr={[1, 2]}>
        <color attach="background" args={["#0d1117"]} />
        <fog attach="fog" args={["#0d1117", 9, 20]} />
        <Clock />
        <Rig />
      </Canvas>
      <Legend />
      <span className="hint">drag to orbit · scroll to zoom</span>
    </>
  );
}
