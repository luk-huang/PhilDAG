// frontend/src/components/GraphView.tsx
import { useEffect, useMemo, useRef } from 'react';
import { select, type Selection } from 'd3-selection';
import { drag } from 'd3-drag';
import { zoom } from 'd3-zoom';
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force';

type Quote = {
  page: number;
  text: string;
};

type Artifact = {
  id: number;
  name: string;
  author: string;
  tile: string;
  year: string;
};

type Claim = {
  id: number;
  artifact: Artifact;
  desc: string;
  quotes?: Quote[];
  citations?: string[];
};

type Argument = {
  id: number;
  premise: Claim[];
  conclusion: Claim;
  desc: string;
};

export type GraphData = {
  claims?: Claim[];
  arguments?: Argument[];
};

type ClaimRole = 'premise' | 'conclusion' | 'both' | 'isolated';
type NodeRole = ClaimRole | 'argument';

type SimNode = SimulationNodeDatum & {
  id: string;
  label: string;
  role: NodeRole;
  radius: number;
  fill: string;
  stroke: string;
  initialX: number;
};

type SimLink = SimulationLinkDatum<SimNode> & {
  source: string;
  target: string;
};

type PreparedGraph = {
  nodes: SimNode[];
  links: SimLink[];
};

type Props = {
  graph: GraphData | null;
};

const ROLE_STYLES: Record<NodeRole, { fill: string; stroke: string; radius: number; layer: number }> = {
  premise: { fill: '#ecf2ff', stroke: 'rgba(174, 191, 255, 0.85)', radius: 80, layer: -220 },
  both: { fill: '#f4f1ff', stroke: 'rgba(194, 180, 255, 0.8)', radius: 84, layer: -60 },
  conclusion: { fill: '#ecfdf5', stroke: 'rgba(168, 217, 196, 0.85)', radius: 82, layer: 220 },
  isolated: { fill: '#f9f6ff', stroke: 'rgba(208, 198, 255, 0.75)', radius: 70, layer: 40 },
  argument: { fill: '#fff7ea', stroke: 'rgba(234, 208, 153, 0.85)', radius: 88, layer: 60 },
};

export function GraphView({ graph }: Props) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const prepared = useMemo(() => (graph ? buildSimulationGraph(graph) : null), [graph]);

  useEffect(() => {
    if (!prepared) return;

    const wrapper = wrapperRef.current;
    const svgElement = svgRef.current;
    if (!wrapper || !svgElement) return;

    const width = wrapper.clientWidth || 960;
    const height = wrapper.clientHeight || 520;

    const svg = select(svgElement);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `${-width / 2} ${-height / 2} ${width} ${height}`);

    const defs = svg.append('defs');
    defs
      .append('marker')
      .attr('id', 'graph-arrowhead')
      .attr('viewBox', '0 -6 12 12')
      .attr('refX', 12)
      .attr('refY', 0)
      .attr('markerWidth', 12)
      .attr('markerHeight', 12)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-6L12,0L0,6')
      .attr('fill', 'rgba(176, 184, 214, 0.8)');

    const content = svg.append('g').attr('class', 'graph-content');

    const zoomBehaviour = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 1.8])
      .on('zoom', (event) => {
        content.attr('transform', event.transform);
      });

    svg.call(zoomBehaviour as any);

    const simulation = forceSimulation(prepared.nodes)
      .force(
        'link',
        forceLink<SimNode, SimLink>(prepared.links)
          .id((d) => d.id)
          .distance(320)
          .strength(0.9),
      )
      .force('charge', forceManyBody<SimNode>().strength(-380))
      .force('collision', forceCollide<SimNode>().radius((d) => d.radius + 16))
      .force('center', forceCenter<SimNode>(0, 20))
      .force('x', forceX<SimNode>().x((d) => d.initialX).strength(0.2))
      .force('y', forceY<SimNode>().y((d) => ROLE_STYLES[d.role].layer).strength(0.6));

    const links = content
      .append('g')
      .attr('class', 'graph-links')
      .selectAll('path')
      .data(prepared.links)
      .enter()
      .append('path')
      .attr('class', 'graph-link')
      .attr('marker-end', 'url(#graph-arrowhead)');

    const nodes = content
      .append('g')
      .attr('class', 'graph-nodes')
      .selectAll('g')
      .data(prepared.nodes)
      .enter()
      .append('g')
      .attr('class', (d) => `graph-node graph-node-${d.role}`)
      .call(
        drag<SVGGElement, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }),
      );

    nodes
      .append('circle')
      .attr('r', (d) => d.radius)
      .attr('fill', (d) => d.fill)
      .attr('stroke', (d) => d.stroke)
      .attr('stroke-width', 3);

    nodes
      .append('text')
      .attr('class', 'graph-node-label')
      .each(function (d) {
        const text = select(this);
        const lines = wrapLabel(d.label);
        lines.forEach((line, index) => {
          text
            .append('tspan')
            .attr('x', 0)
            .attr('dy', index === 0 ? '0.35em' : '1.1em')
            .text(line);
        });
      });

    simulation.on('tick', () => {
      links.attr('d', (link) => computeLinkPath(link));
      nodes.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [prepared]);

  if (!prepared) {
    return <div className="graph-placeholder">Your graph will appear here.</div>;
  }

  return (
    <div className="graph-container" ref={wrapperRef}>
      <svg ref={svgRef} className="graph-svg" role="presentation" />
    </div>
  );
}

function buildSimulationGraph(graph: GraphData): PreparedGraph {
  const argumentsList = graph.arguments ?? [];
  const explicitClaims = graph.claims ?? [];

  const claimMap = new Map<number, Claim>();
  explicitClaims.forEach((claim) => {
    if (claim?.id !== undefined && claim !== null) {
      claimMap.set(claim.id, claim);
    }
  });

  const claimRoles = new Map<number, ClaimRole>();

  argumentsList.forEach((argument) => {
    argument.premise.forEach((premise) => {
      claimMap.set(premise.id, premise);
      const previous = claimRoles.get(premise.id);
      if (previous === 'conclusion') {
        claimRoles.set(premise.id, 'both');
      } else if (!previous || previous === 'isolated') {
        claimRoles.set(premise.id, 'premise');
      }
    });

    claimMap.set(argument.conclusion.id, argument.conclusion);
    const conclusionRole = claimRoles.get(argument.conclusion.id);
    if (conclusionRole === 'premise') {
      claimRoles.set(argument.conclusion.id, 'both');
    } else if (!conclusionRole || conclusionRole === 'isolated') {
      claimRoles.set(argument.conclusion.id, 'conclusion');
    }
  });

  claimMap.forEach((claim) => {
    if (!claimRoles.has(claim.id)) {
      claimRoles.set(claim.id, 'isolated');
    }
  });

  const nodes: SimNode[] = [];
  const links: SimLink[] = [];
  const registered = new Set<string>();

  const roleCounters: Record<NodeRole, number> = {
    premise: 0,
    both: 0,
    conclusion: 0,
    isolated: 0,
    argument: 0,
  };

  const roleTotals: Record<NodeRole, number> = {
    premise: 0,
    both: 0,
    conclusion: 0,
    isolated: 0,
    argument: argumentsList.length,
  };

  claimMap.forEach((claim) => {
    const role = claimRoles.get(claim.id) ?? 'isolated';
    roleTotals[role] += 1;
  });

  const nextInitialX = (role: NodeRole) => {
    const total = Math.max(roleTotals[role], 1);
    const index = roleCounters[role]++;
    const spacing = 230;
    return (index - (total - 1) / 2) * spacing;
  };

  claimMap.forEach((claim) => {
    const role = claimRoles.get(claim.id) ?? 'isolated';
    const nodeId = claimNodeId(claim.id);
    if (registered.has(nodeId)) return;
    registered.add(nodeId);
    const style = ROLE_STYLES[role];
    nodes.push({
      id: nodeId,
      label: claim.desc,
      role,
      radius: style.radius,
      fill: style.fill,
      stroke: style.stroke,
      initialX: nextInitialX(role),
    });
  });

  argumentsList.forEach((argument) => {
    const nodeId = argumentNodeId(argument.id);
    if (!registered.has(nodeId)) {
      registered.add(nodeId);
      const style = ROLE_STYLES.argument;
      nodes.push({
        id: nodeId,
        label: argument.desc,
        role: 'argument',
        radius: style.radius,
        fill: style.fill,
        stroke: style.stroke,
        initialX: nextInitialX('argument'),
      });
    }

    argument.premise.forEach((premise) => {
      const sourceId = claimNodeId(premise.id);
      links.push({ source: sourceId, target: nodeId });
    });

    links.push({ source: nodeId, target: claimNodeId(argument.conclusion.id) });
  });

  return { nodes, links };
}

function claimNodeId(id: number): string {
  return `claim-${id}`;
}

function argumentNodeId(id: number): string {
  return `argument-${id}`;
}

function wrapLabel(label: string, maxChars = 26): string[] {
  const words = label.trim().split(/\s+/);
  const lines: string[] = [];
  let current = '';

  words.forEach((word) => {
    const tentative = current ? `${current} ${word}` : word;
    if (tentative.length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = tentative;
    }
  });

  if (current) {
    lines.push(current);
  }

  if (!lines.length) {
    lines.push(label);
  }

  return lines;
}

function computeLinkPath(link: SimulationLinkDatum<SimNode>): string {
  const source = link.source as SimNode;
  const target = link.target as SimNode;
  if (!source || !target) {
    return 'M0,0L0,0';
  }

  const dx = (target.x ?? 0) - (source.x ?? 0);
  const dy = (target.y ?? 0) - (source.y ?? 0);
  const distance = Math.sqrt(dx * dx + dy * dy) || 1;

  const sx = (source.x ?? 0) + (dx * (source.radius - 6)) / distance;
  const sy = (source.y ?? 0) + (dy * (source.radius - 6)) / distance;
  const tx = (target.x ?? 0) - (dx * (target.radius + 8)) / distance;
  const ty = (target.y ?? 0) - (dy * (target.radius + 8)) / distance;

  return `M${sx},${sy}L${tx},${ty}`;
}
