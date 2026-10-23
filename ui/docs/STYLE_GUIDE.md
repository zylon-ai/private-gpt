# PrivateGPT Workbench Style Guide

This guide defines the visual and UX direction for PrivateGPT Workbench. It complements the product requirements in [`docs/PRD.md`](./PRD.md).

The goal is to keep the implementation simple while aligning the demonstrator with the public-facing PrivateGPT/Zylon visual language.

## Reference Assets

Use these repo-local images as visual references:

| Reference | File | Purpose |
| --- | --- | --- |
| Primary chat layout | [`../references/primary-chat-layout.png`](../references/primary-chat-layout.png) | Main reference for sidebar, chat layout, message bubbles, and composer placement. |
| Search overlay | [`../references/search-overlay.png`](../references/search-overlay.png) | Reference for modal/search overlays, large glass panels, tabs/chips, and filtered result lists. |
| Chat tools composer | [`../references/chat-tools-composer.png`](../references/chat-tools-composer.png) | Reference for the chat composer, file/tool controls, message density, and chat bubble treatment. |
| Context knowledge base | [`../references/context-knowledge-base.png`](../references/context-knowledge-base.png) | Reference for Context rows, source lists, file badges, overflow menus, and glass list surfaces. |

The PrivateGPT logo should be embedded directly in `./index.html` as inline SVG so the page does not depend on an external logo asset at runtime.

## Design Intent

PrivateGPT Workbench should feel like a polished local AI workspace, not a generic admin console and not a marketing landing page.

The interface should communicate:

- Local-first AI utility.
- Technical power without intimidating non-technical users.
- A premium, public-brand-aligned experience.
- Simplicity and directness.

The product surface remains:

```text
Sidebar
  Context
  New Chat
  Chat list
  API Debugger
  Settings
  GitHub
  Not for Production

Main
  Context screen
  or
  API Debugger screen
  or
  Settings screen
  or
  Chat screen
```

## Visual Language

Use a dark, atmospheric workspace:

- Deep navy, charcoal, black, muted amber, and subtle purple/brown tones.
- Radial or blended gradients across the full viewport.
- A fine grain/noise texture over the background.
- Frosted glass panels with blur, light borders, and soft shadows.
- White primary text and muted gray secondary text.
- Small blue/orange/purple gradient accents inspired by the PrivateGPT/Zylon orb.

Avoid:

- Flat admin-dashboard gray.
- Marketing hero sections.
- Large decorative cards unrelated to function.
- Bright white page backgrounds.
- Excessive purple/blue gradient dominance.
- Heavy component-library appearance.

## Layout

### App Shell

Follow the structure in `primary-chat-layout.png`.

Recommended dimensions:

- Sidebar width: `286px` (`--sidebar` variable).
- Main content max width for chat: `860px`.
- Composer width: aligned to chat content width.
- Full viewport height.
- No separate header bar unless needed.

The background should remain visible around and through glass surfaces.

### Sidebar

The sidebar should be a vertical frosted-glass panel.

Content:

```text
PrivateGPT logo

Context
New Chat

Chats
  Chat title
  Chat title
  Chat title

API Debugger
Settings
GitHub
Not for Production
```

Requirements:

- Place the PrivateGPT logo at the top-left. In the single-file app, use the inline embedded SVG.
- Keep navigation compact and readable.
- Active item uses a brighter glass state.
- Chat rows should truncate long titles.
- No projects, folders, or chat grouping.
- Place API Debugger above Settings at the bottom of the sidebar.
- Place Settings below API Debugger in the bottom sidebar group.
- Place the GitHub repository widget below Settings.
- Place the Not for Production disclosure below the GitHub widget.
- Avoid making the sidebar feel like an enterprise admin menu.
- The chat list fills all available vertical space between the nav buttons and the bottom group, using `flex: 1 1 0` and `min-height: 0`.
- Apply scroll-aware top/bottom fade masks to the chat list using CSS `mask-image` with `--fade-top-stop`/`--fade-bot-stop` custom properties, updated on scroll.

### Main Chat

The chat view should be centered and spacious.

Message behavior:

- User messages align right.
- Assistant messages align left.
- Bubbles use translucent glass fills.
- Assistant messages may include a small gradient orb/avatar.
- User messages may use a subtle avatar or label.
- Citations should appear inline as compact superscript-like markers or small chips.
- Tool activity should appear as subdued inline status blocks, not large cards.
- Apply scroll-aware top/bottom fade masks to the messages list using the same `mask-image` pattern as the chat list.

### Composer

Use `chat-tools-composer.png` as the main composer reference.

The composer should include:

- Compact two-row glass input inspired by the reference composer. Copy only its visual treatment and interaction patterns; do not introduce labels, modes, context syntax, or capabilities that PrivateGPT does not implement.
- A text area above a single toolbar row.
- Left-side toolbar controls for a consolidated add/actions menu, searchable model and reasoning-effort selection, and model refresh.
- Right-side circular send control.
- Use a slightly taller composer input and 32px toolbar controls with medium-weight labels so the primary input remains visually substantial without becoming a large card.
- Composer controls use self-contained inline SVGs so global icon hydration and sizing cannot distort them.
- The add/actions control starts as a centered circular plus button and smoothly expands to reveal a short “Add” label on hover or keyboard focus. Its menu contains file attachment first, followed by the existing PrivateGPT context/tool configuration.
- Toolbar icons use restrained hover motion: refresh rotates, send lifts, and dropdown chevrons respond to hover/open state.
- Composer dropdowns morph outward from their trigger using a spring-like scale and corner-radius transition, then reverse into the trigger when closed.
- The model effort rail reveals after the main model list with a short lateral clip transition, while model and effort rows enter with a restrained stagger. Filtering repeats the row transition so list changes remain legible.
- Respect `prefers-reduced-motion` by disabling composer menu, rail, and row animations.
- Floating composer menus must clamp their width, height, and horizontal position to the current viewport. Their internal lists scroll without allowing the panel itself to render beyond the window.
- File attachment uploads to the active code-execution session when Code Execution is enabled; otherwise it ingests files into the configured Documents collection.
- The model dropdown includes a focused search field and filters loaded models by display name.
- The model dropdown uses a two-column layout: searchable models on the left and a reasoning-effort rail on the right. Effort choices are None, Low, Medium, High, Max, and XHigh, with unsupported model capabilities disabled.
- Reasoning effort replaces the standalone Thinking composer button and is stored per chat.

The `Tools` control should expose chat-specific toggles:

- Documents
- Web
- Databases
- MCP
- Skills
- Custom Tools

Selected context items can appear as compact chips.

## Context Screen

The Context screen defines what the assistant can access. It should look like a source manager, not a settings page.

Sections:

```text
Documents
Databases
Web
MCP
Skills
Custom Tools
```

Use `context-knowledge-base.png` as the reference for:

- Glass list rows.
- File/source icons.
- Type badges.
- Status metadata.
- Overflow menus.
- Nested or grouped rows where useful.

### Documents

Rows should show:

- Icon.
- Name.
- Collection.
- File type badge, such as `PDF`, `DOCX`, `CSV`, `HTML`.
- Status, such as `Indexed`, `Processing`, `Failed`.
- Overflow menu: Preview, Search, Delete.

The Collection field is in Settings, not in the Documents panel. It applies globally to all document operations.

### Databases

Rows should show:

- Database icon.
- Friendly name.
- Host or short connection label.
- Schema/table metadata if available.
- Overflow menu: Edit, Test, Delete.

### Web, MCP, Skills, Custom Tools

Use the same row/card language:

- Name.
- Description or provider.
- Configuration summary.
- Overflow actions.

The Web section is informational only. Do not collect provider credentials in Workbench; web provider and API key configuration belongs in the PrivateGPT backend.

Custom Tools should not be hidden behind an "Advanced" label. They are a first-class Context section.

## Settings Screen

Settings owns Workbench-level connection configuration:

- PrivateGPT API base URL.
- Optional HTTP Basic auth (username and password fields displayed side by side).
- Optional system prompt.
- Optional workspace instructions for the current branded experience.
- Use citations toggle.
- **Collection** — the active document collection name used for all document operations and chat requests. This belongs in Settings, not in the Documents panel.
- Appearance controls for brand copy, palette, and optional visible sections.
- A control to rerun onboarding.
- Test API and Save buttons.
- Clear local data.

## Onboarding Overlay

The first-run onboarding should feel like a guided setup sheet rather than a wizard from an enterprise admin console.

- Present it as a large glass overlay above the existing app shell so the user can feel that they are configuring the actual workspace.
- Step 1 should emphasize clarity and confidence: URL, auth, collection, then a simple checklist showing whether models, collection, and skills responded.
- Step 2 should feel lighter and optional: a prompt-driven generator plus editable result form on one side and a live preview tile on the other.
- The optional customization step should use the same visual language as Settings so the user understands both surfaces write to the same variables.
- GitHub / Zylon references are part of the demo identity and should not be removable through appearance customization.

## API Debugger

API Debugger appears as a sidebar destination above Settings.

It is session-level, live-only, and ephemeral.

Recommended layout:

```text
Timeline list | Event detail panel
```

Use the same glass styling, but make it denser and more technical than Chat.

API Debugger should show:

- API requests.
- API responses.
- Errors.
- Redacted request headers.

API Debugger events must not persist across page reloads.

Both the timeline panel and the detail panel must have `min-width: 0` so long URLs truncate instead of overflowing the panel.

## Not For Production Disclosure

The sidebar includes a compact `Not for Production` button below the GitHub widget.

Visual treatment:

- Same sidebar row rhythm as the GitHub widget.
- Slight warm/danger tint so it reads as an important disclosure without looking like an error.
- Info icon plus text label.

Clicking it opens a glass-style modal. The title should be:

```text
This demonstrator is not intended for Production use
```

Keep the body concise, with four merged bullets covering browser `localStorage` secrets, lack of access control, visible debugger data, and browser-executed custom tools. End with links to Zylon (`https://zylon.ai`) and the demo booking page (`https://cal.com/zylon/demo?source=privategptui`).

## GitHub Widget

The sidebar includes a clickable GitHub widget below Settings.

Requirements:

- Link to `https://github.com/zylon-ai/private-gpt`.
- Show the GitHub icon, label, star icon, and live star count when available.
- If the star fetch fails, keep the widget usable and show a neutral `Stars` label.

## URL Hash Navigation

The app uses hash-based navigation so reloading restores the current view.

Hash format:

| Hash | View |
| --- | --- |
| `#context/documents` | Context — Documents tab |
| `#context/databases` | Context — Databases tab |
| `#context/web` | Context — Web tab |
| `#context/mcp` | Context — MCP tab |
| `#context/skills` | Context — Skills tab |
| `#context/customTools` | Context — Custom Tools tab |
| `#settings` | Settings screen |
| `#apiDebugger` | API Debugger screen |
| `#chat/{chatId}` | Specific chat by ID |

`syncHash()` calls `history.replaceState` at the end of every `render()` call. `restoreFromHash()` runs at startup before the first render and on `hashchange` events for browser back/forward support.

## Components

### Glass Surface

All main panels share one glass treatment via a shared CSS selector group:

```css
.settings-card,
.context-panel,
.debug-panel,
.composer,
.message-bubble,
.menu-panel,
.model-dropdown,
.modal-card {
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.055));
  border: 1px solid var(--border);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);
}
```

Floating panels (modal card, tools menu, model dropdown) override with a light translucent frost, very heavy blur, heavier at the bottom:

```css
.modal-card,
.menu-panel,
.model-dropdown {
  background: linear-gradient(to bottom,
    rgba(255, 255, 255, 0.14) 0%,
    rgba(255, 255, 255, 0.26) 100%
  );
  backdrop-filter: blur(72px) saturate(1.6);
  -webkit-backdrop-filter: blur(72px) saturate(1.6);
}
```

The white-glass tint stays light (translucent, not opaque); the `blur(72px)` defocuses the background enough to make text readable; the gradient is heavier at the bottom for visual grounding.

The modal backdrop itself uses a moderate blur:

```css
.modal-backdrop {
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}
```

### Background

```css
:root {
  --bg: #090b12;
  --text: #f7f7fb;
  --muted: rgba(255, 255, 255, 0.58);
  --muted-strong: rgba(255, 255, 255, 0.74);
  --faint: rgba(255, 255, 255, 0.36);
  --glass: rgba(255, 255, 255, 0.1);
  --glass-soft: rgba(255, 255, 255, 0.07);
  --glass-strong: rgba(255, 255, 255, 0.16);
  --border: rgba(255, 255, 255, 0.24);
  --border-soft: rgba(255, 255, 255, 0.14);
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
  --danger: #ff9f9f;
  --ok: #91e8bd;
  --warn: #ffd18a;
  --accent: #f0a247;
  --blue: #70b7ff;
  --radius-xl: 32px;
  --radius-lg: 24px;
  --radius-md: 16px;
  --radius-sm: 10px;
  --sidebar: 286px;
}

body {
  background:
    radial-gradient(circle at 17% 78%, rgba(36, 125, 190, 0.58), transparent 36%),
    radial-gradient(circle at 83% 16%, rgba(212, 148, 63, 0.38), transparent 31%),
    radial-gradient(circle at 53% 42%, rgba(92, 50, 116, 0.34), transparent 40%),
    linear-gradient(135deg, #081427 0%, #15121c 42%, #130b05 100%);
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.24;
  background-image:
    radial-gradient(rgba(255,255,255,0.18) 0.65px, transparent 0.65px);
  background-size: 3px 3px;
  mix-blend-mode: overlay;
}
```

### Model Selector

Replace the native `<select>` with a custom glass dropdown:

- A `ghost-button` showing a CPU icon, the current model name (truncated), and an animated chevron.
- Clicking opens a `.model-dropdown` panel positioned above the composer.
- The dropdown lists all models as `.model-option` rows with a checkmark on the active selection.
- Close on outside click or Escape.
- The `.model-dropdown` is in the shared glass group and gets the stronger frost override.

### Tools Menu

The Tools popup (`.menu-panel`) is structured with section groups:

```text
Knowledge
  Documents  [toggle]
  Web        [toggle]

Databases    [group title]
  item       [toggle]

MCP          [group title]
  ...

Skills       [group title]
  ...

Custom Tools [group title]
  ...
```

Each group uses `.menu-group` and `.menu-group-title` (small uppercase label). Toggle rows inside the menu are borderless with hover highlight only, distinguishing them from Settings card toggle rows which retain a border.

### Toggle Switches

All `input[type="checkbox"]` elements are styled as custom pill toggles — no native appearance:

- Pill track: `34 × 20px`, rounded.
- Knob: `12px` circle, vertically centered.
- Off state: muted white track and knob.
- On state: blue tint track (`rgba(80, 150, 255, 0.32)`) with a bright knob (`rgba(150, 205, 255, 0.96)`).
- Smooth `left` transition on the knob (not `transform: translateX`).

### Thinking Button

The Extended Thinking toggle in the composer uses:

- A `ghost-button` with a `zap` icon and the label `Thinking`.
- Inactive: standard ghost button appearance with purple hover hint.
- Active: purple border, purple-tinted background, soft glow, and purple-tinted icon.

```css
.thinking-chip.active {
  border-color: rgba(130, 80, 230, 0.55);
  background: rgba(100, 55, 200, 0.18);
  color: rgba(200, 165, 255, 0.95);
  box-shadow: 0 0 14px rgba(120, 70, 220, 0.18);
}
```

### Buttons And Chips

Buttons and chips should feel like part of the glass environment:

- Thin border.
- Translucent fill.
- Slightly brighter hover.
- Clear active state with `scale(0.97)` press feedback.
- Icons where useful.

Use icons for:

- Send.
- Search.
- Files/collection.
- Tools.
- Database.
- More menu.
- Delete.
- Refresh/test.
- CPU (model selector).
- Chevron-down (model selector dropdown indicator).
- Check (selected item in model dropdown).
- Zap (thinking toggle).

If no icon library is used, inline SVGs are acceptable, but keep them minimal.

### Menus And Overlays

Use `search-overlay.png` as reference.

Requirements:

- Glass overlay panel with the stronger frost treatment.
- Strong border.
- Soft shadow.
- Large readable input for search-like overlays.
- Compact pill filters where needed.
- Results as clean rows.

Both `.menu-panel` and `.model-dropdown` share the same glass + stronger-frost override as `.modal-card`.

### Scroll Fades

Apply a top/bottom scroll-aware fade mask to any scrollable list that may overflow:

```css
.scrollable-wrap {
  --fade-top-stop: 0px;
  --fade-bot-stop: 0px;
  mask-image: linear-gradient(
    to bottom,
    transparent 0,
    black var(--fade-top-stop),
    black calc(100% - var(--fade-bot-stop)),
    transparent 100%
  );
}
```

Update `--fade-top-stop` and `--fade-bot-stop` via JavaScript on scroll. Applied to:

- `.chat-list-wrap` — sidebar chat list (22px stops).
- `.messages` — main message list (28px stops).

## Typography

Use a modern sans-serif stack:

```css
font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
```

Guidelines:

- Primary text: high contrast white.
- Secondary text: muted white/gray.
- Avoid oversized hero typography.
- Chat text should be comfortably readable.
- Context/debugger text can be denser.
- Do not use negative letter spacing.
- Do not scale font sizes with viewport width.

## Responsive Behavior

Desktop is the primary target for v1.

Minimum behavior:

- At narrow widths, sidebar can collapse or become an overlay.
- Chat composer remains usable.
- Text must not overflow buttons, chips, or rows.
- Context rows should wrap metadata rather than clipping important labels.

## Implementation Constraints

The visual system should not require a frontend framework.

Preferred implementation:

- Single static `./index.html`.
- Vanilla CSS variables.
- Small reusable CSS classes.
- Repeated HTML templates are acceptable.
- No design system extraction required.

The styling should support the PRD's simplicity requirement: polished enough for a public-facing demo, but not architected like a long-term product frontend.
