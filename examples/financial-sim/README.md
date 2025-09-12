# Financial Simulation - Chart of Accounts as Virtual Hardware

This example demonstrates modeling a financial Chart of Accounts (CoA) as virtual hardware architecture, leveraging llama.cpp to simulate, route, and reason over financial operations.

## Concept

- **Accounts as Hardware Nodes**: Each account (Assets, Liabilities, Income, Expenses, Equity) is treated as a hardware node or pin
- **Transactions as Signal Routing**: Money transfers are modeled as electrical signals flowing through circuits
- **Account Groups as Subsystems**: Account classifications map to hardware modules/sub-chips
- **Business Rules as Circuit Logic**: Double-entry accounting becomes conservation laws, validations become electrical laws

## Usage

```bash
# Basic simulation with built-in chart of accounts
./llama-financial-sim -m model.gguf

# Load custom chart of accounts
./llama-financial-sim -m model.gguf --coa accounts.json

# Interactive transaction simulation
./llama-financial-sim -m model.gguf --interactive

# Trace specific transaction flows
./llama-financial-sim -m model.gguf --trace "Transfer $1000 from Cash to Equipment"
```

## Features

- Chart of Accounts hierarchy modeling
- Double-entry transaction validation
- Hardware-style signal flow tracing
- LLM-powered financial reasoning
- Interactive transaction simulation
- Account balance monitoring
- Business rule validation

## Example Queries

- "Route $X from Account A to Account B. What downstream signals change?"
- "If a surge happens in Revenue, which accounts will be impacted?"
- "Trace the propagation of this transaction through the financial circuit"
- "Validate this journal entry using circuit laws"