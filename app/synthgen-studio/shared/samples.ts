/**
 * Curated sample objects. Each has a source photo in the raw_inputs volume (samples/) and, per workspace,
 * a finished pipeline run mapped through the SAMPLE_RUN_IDS env var ("key:runId,key:runId").
 * Photos are from Wikimedia Commons; see the README for full credits.
 */
export interface SampleDef {
  key: string;
  label: string;
  domain: string;
  /** Path relative to the raw_inputs volume. */
  image: string;
  className: string;
  credit: string;
}

export const SAMPLES: SampleDef[] = [
  {
    key: 'traffic_cone',
    label: 'Traffic cone',
    domain: 'Field obstacle',
    image: 'samples/traffic_cone.jpg',
    className: 'traffic_cone',
    credit: "Filo gèn', CC BY-SA 4.0, via Wikimedia Commons",
  },
  {
    key: 'tractor_tire',
    label: 'Tractor tire',
    domain: 'Field obstacle',
    image: 'samples/tractor_tire.jpg',
    className: 'tractor_tire',
    credit: 'Evelyn Simak, CC BY-SA 2.0, via Wikimedia Commons',
  },
  {
    key: 'bearing',
    label: 'Ball bearing',
    domain: 'Manufacturing part',
    image: 'samples/bearing_fag_6304.jpg',
    className: 'bearing',
    credit: 'R. Henrik Nilsson, CC BY 4.0, via Wikimedia Commons',
  },
];
