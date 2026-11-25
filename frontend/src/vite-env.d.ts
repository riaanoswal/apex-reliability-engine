/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional absolute backend URL. Empty/unset => /api is proxied by the dev server. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
