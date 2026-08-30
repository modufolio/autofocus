import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'
import { fileURLToPath } from 'url'

export default defineConfig({
  plugins: [
    // Declarations from the JSDoc'd source (allowJs) so consumers get types.
    dts({ tsconfigPath: './tsconfig.build.json', entryRoot: 'src', rollupTypes: false }),
  ],
  build: {
    lib: {
      entry: fileURLToPath(new URL('./src/index.js', import.meta.url)),
      formats: ['es'],
      fileName: 'index',
    },
    // Zero runtime dependencies: the WASM module is fetched at runtime from
    // the consuming app's /assets/wasm/autofocus/, never bundled.
    rollupOptions: { external: [] },
    sourcemap: true,
    emptyOutDir: true,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.spec.js'],
  },
})
