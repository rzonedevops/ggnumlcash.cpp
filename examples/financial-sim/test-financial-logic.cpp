#include <iostream>
#include <cassert>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

// Include the chart of accounts code from the main file
enum class AccountType {
    ASSET,      // Input registers/storage
    LIABILITY,  // Output buffers/obligations
    EQUITY,     // Core processing unit
    REVENUE,    // Signal generators/inputs
    EXPENSE     // Signal consumers/outputs
};

struct Account {
    std::string code;           // Pin/node identifier
    std::string name;           // Human readable name
    AccountType type;           // Hardware subsystem type
    double balance;             // Current signal level/voltage
    std::string parent_code;    // Parent node in hierarchy
    std::vector<std::string> children; // Child nodes
    bool is_debit_normal;       // Signal polarity
    
    Account() : code(""), name(""), type(AccountType::ASSET), balance(0.0), parent_code(""), is_debit_normal(true) {}
    
    Account(const std::string& c, const std::string& n, AccountType t, const std::string& parent = "")
        : code(c), name(n), type(t), balance(0.0), parent_code(parent), is_debit_normal(t == AccountType::ASSET || t == AccountType::EXPENSE) {}
};

struct TransactionEntry {
    std::string account_code;   // Target node
    double debit_amount;        // Positive signal flow
    double credit_amount;       // Negative signal flow
    std::string description;    // Signal description
};

struct Transaction {
    std::string id;
    std::string description;
    std::vector<TransactionEntry> entries;
    std::string timestamp;
    
    bool is_balanced() const {
        double total_debits = 0.0, total_credits = 0.0;
        for (const auto& entry : entries) {
            total_debits += entry.debit_amount;
            total_credits += entry.credit_amount;
        }
        return std::abs(total_debits - total_credits) < 0.01;
    }
};

class ChartOfAccounts {
private:
    std::map<std::string, Account> accounts;
    std::vector<Transaction> transaction_log;
    
public:
    void initialize_standard_coa() {
        // Asset Subsystem (Input/Storage)
        add_account("1000", "Assets", AccountType::ASSET);
        add_account("1100", "Current Assets", AccountType::ASSET, "1000");
        add_account("1101", "Cash", AccountType::ASSET, "1100");
        add_account("1102", "Accounts Receivable", AccountType::ASSET, "1100");
        
        // Liability Subsystem (Output/Obligations)
        add_account("2000", "Liabilities", AccountType::LIABILITY);
        add_account("2100", "Current Liabilities", AccountType::LIABILITY, "2000");
        add_account("2101", "Accounts Payable", AccountType::LIABILITY, "2100");
        
        // Equity Subsystem (Core Processing)
        add_account("3000", "Equity", AccountType::EQUITY);
        add_account("3100", "Owner's Equity", AccountType::EQUITY, "3000");
        
        // Revenue Subsystem (Signal Sources)
        add_account("4000", "Revenue", AccountType::REVENUE);
        add_account("4100", "Sales Revenue", AccountType::REVENUE, "4000");
        
        // Expense Subsystem (Signal Sinks)
        add_account("5000", "Expenses", AccountType::EXPENSE);
        add_account("5100", "Operating Expenses", AccountType::EXPENSE, "5000");
        add_account("5101", "Salaries Expense", AccountType::EXPENSE, "5100");
    }
    
    void add_account(const std::string& code, const std::string& name, AccountType type, const std::string& parent = "") {
        accounts[code] = Account(code, name, type, parent);
        if (!parent.empty() && accounts.find(parent) != accounts.end()) {
            accounts[parent].children.push_back(code);
        }
    }
    
    bool account_exists(const std::string& code) const {
        return accounts.find(code) != accounts.end();
    }
    
    Account* get_account(const std::string& code) {
        auto it = accounts.find(code);
        return (it != accounts.end()) ? &it->second : nullptr;
    }
    
    bool process_transaction(const Transaction& transaction) {
        if (!transaction.is_balanced()) {
            std::cerr << "Error: Transaction violates conservation law (not balanced)\n";
            return false;
        }
        
        // Validate all accounts exist
        for (const auto& entry : transaction.entries) {
            if (!account_exists(entry.account_code)) {
                std::cerr << "Error: Account node " << entry.account_code << " not found in circuit\n";
                return false;
            }
        }
        
        // Apply signal routing
        for (const auto& entry : transaction.entries) {
            Account* account = get_account(entry.account_code);
            if (account->is_debit_normal) {
                account->balance += entry.debit_amount - entry.credit_amount;
            } else {
                account->balance += entry.credit_amount - entry.debit_amount;
            }
        }
        
        // Log the transaction
        transaction_log.push_back(transaction);
        return true;
    }
    
    double get_balance(const std::string& code) const {
        auto it = accounts.find(code);
        return (it != accounts.end()) ? it->second.balance : 0.0;
    }
    
    size_t get_transaction_count() const {
        return transaction_log.size();
    }
};

// Test functions
void test_chart_of_accounts_initialization() {
    std::cout << "Testing Chart of Accounts initialization...\n";
    
    ChartOfAccounts coa;
    coa.initialize_standard_coa();
    
    // Test that key accounts exist
    assert(coa.account_exists("1101")); // Cash
    assert(coa.account_exists("2101")); // Accounts Payable
    assert(coa.account_exists("3100")); // Owner's Equity
    assert(coa.account_exists("4100")); // Sales Revenue
    assert(coa.account_exists("5101")); // Salaries Expense
    
    std::cout << "✓ Chart of Accounts initialized correctly\n";
}

void test_transaction_balancing() {
    std::cout << "Testing transaction balancing (conservation laws)...\n";
    
    // Test balanced transaction
    Transaction balanced_tx;
    balanced_tx.id = "TX001";
    balanced_tx.description = "Test balanced transaction";
    balanced_tx.entries = {
        {"1101", 1000.0, 0.0, "Cash debit"},
        {"3100", 0.0, 1000.0, "Equity credit"}
    };
    
    assert(balanced_tx.is_balanced());
    std::cout << "✓ Balanced transaction validation works\n";
    
    // Test unbalanced transaction
    Transaction unbalanced_tx;
    unbalanced_tx.id = "TX002";
    unbalanced_tx.description = "Test unbalanced transaction";
    unbalanced_tx.entries = {
        {"1101", 1000.0, 0.0, "Cash debit"},
        {"3100", 0.0, 500.0, "Equity credit - UNBALANCED"}
    };
    
    assert(!unbalanced_tx.is_balanced());
    std::cout << "✓ Unbalanced transaction detection works\n";
}

void test_signal_routing() {
    std::cout << "Testing signal routing through financial circuit...\n";
    
    ChartOfAccounts coa;
    coa.initialize_standard_coa();
    
    // Create a transaction: Initial investment
    Transaction investment;
    investment.id = "TX001";
    investment.description = "Initial capital investment";
    investment.entries = {
        {"1101", 10000.0, 0.0, "Cash received"},  // Debit Cash (Asset)
        {"3100", 0.0, 10000.0, "Owner investment"} // Credit Owner's Equity
    };
    
    assert(coa.process_transaction(investment));
    assert(coa.get_balance("1101") == 10000.0); // Cash should have $10,000
    assert(coa.get_balance("3100") == 10000.0); // Owner's Equity should have $10,000
    
    std::cout << "✓ Investment transaction processed correctly\n";
    
    // Create a transaction: Purchase equipment
    Transaction purchase;
    purchase.id = "TX002";
    purchase.description = "Purchase office equipment";
    purchase.entries = {
        {"1102", 5000.0, 0.0, "Equipment purchased"}, // Debit Equipment (Asset)
        {"1101", 0.0, 5000.0, "Cash paid"}            // Credit Cash (Asset)
    };
    
    // Add equipment account first
    coa.add_account("1102", "Equipment", AccountType::ASSET, "1100");
    
    assert(coa.process_transaction(purchase));
    assert(coa.get_balance("1101") == 5000.0);  // Cash reduced to $5,000
    assert(coa.get_balance("1102") == 5000.0);  // Equipment now $5,000
    
    std::cout << "✓ Purchase transaction processed correctly\n";
    
    // Create a transaction: Earn revenue
    Transaction revenue;
    revenue.id = "TX003";
    revenue.description = "Service revenue earned";
    revenue.entries = {
        {"1101", 2000.0, 0.0, "Cash received"},    // Debit Cash
        {"4100", 0.0, 2000.0, "Service revenue"}   // Credit Revenue
    };
    
    assert(coa.process_transaction(revenue));
    assert(coa.get_balance("1101") == 7000.0);  // Cash now $7,000
    assert(coa.get_balance("4100") == 2000.0);  // Revenue $2,000
    
    std::cout << "✓ Revenue transaction processed correctly\n";
    
    // Create a transaction: Pay salary expense
    Transaction expense;
    expense.id = "TX004";
    expense.description = "Pay employee salaries";
    expense.entries = {
        {"5101", 1500.0, 0.0, "Salary expense"},   // Debit Expense
        {"1101", 0.0, 1500.0, "Cash paid"}         // Credit Cash
    };
    
    assert(coa.process_transaction(expense));
    assert(coa.get_balance("1101") == 5500.0);  // Cash now $5,500
    assert(coa.get_balance("5101") == 1500.0);  // Salary Expense $1,500
    
    std::cout << "✓ Expense transaction processed correctly\n";
    
    assert(coa.get_transaction_count() == 4);
    std::cout << "✓ All " << coa.get_transaction_count() << " transactions logged\n";
}

void test_hardware_analogy_terminology() {
    std::cout << "Testing hardware analogy concepts...\n";
    
    ChartOfAccounts coa;
    coa.initialize_standard_coa();
    
    // Test that different account types have correct signal polarity
    Account* cash = coa.get_account("1101");          // Asset - debit normal
    Account* payable = coa.get_account("2101");       // Liability - credit normal
    Account* equity = coa.get_account("3100");        // Equity - credit normal
    Account* revenue = coa.get_account("4100");       // Revenue - credit normal
    Account* expense = coa.get_account("5101");       // Expense - debit normal
    
    assert(cash->is_debit_normal == true);      // Assets increase with positive signals
    assert(payable->is_debit_normal == false);  // Liabilities increase with negative signals
    assert(equity->is_debit_normal == false);   // Equity increases with negative signals
    assert(revenue->is_debit_normal == false);  // Revenue increases with negative signals
    assert(expense->is_debit_normal == true);   // Expenses increase with positive signals
    
    std::cout << "✓ Signal polarity correctly configured for hardware analogy\n";
    std::cout << "  - Assets & Expenses: Positive signal polarity (debit normal)\n";
    std::cout << "  - Liabilities, Equity & Revenue: Negative signal polarity (credit normal)\n";
}

void run_all_tests() {
    std::cout << "=== FINANCIAL CIRCUIT SIMULATOR TESTS ===\n\n";
    
    test_chart_of_accounts_initialization();
    std::cout << "\n";
    
    test_transaction_balancing();
    std::cout << "\n";
    
    test_signal_routing();
    std::cout << "\n";
    
    test_hardware_analogy_terminology();
    std::cout << "\n";
    
    std::cout << "=== ALL TESTS PASSED ===\n";
    std::cout << "Financial circuit simulation logic is working correctly!\n";
    std::cout << "Ready for LLM integration and interactive simulation.\n";
}

int main() {
    try {
        run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}