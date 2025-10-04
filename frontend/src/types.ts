export interface Quote {
  id?: number;
  page: number;
  text: string;
  statement_id?: number;
  statement?: Statement;
}

export interface Artifact {
  id: number;
  name: string;
  author: string;
  title: string;
  year: string;
  statements?: Statement[];
}

export interface Statement {
  id: number;
  statement: string;
  artifact: Artifact[];
  citations: Quote[];
  arguments_as_premise?: Argument[];
  conclusions_for?: Argument[];
}

export interface Argument {
  id: number;
  desc: string;
  premise: Statement[];
  conclusion: Statement;
}

export interface GraphData {
  statements?: Statement[];
  arguments?: Argument[];
}

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskPhilRequest {
  question: string;
  deepdag?: boolean;
  graph?: GraphData;
  history?: ConversationTurn[];
}

export interface AskPhilResponse {
  answer: string;
  subgraph: GraphData;
}
