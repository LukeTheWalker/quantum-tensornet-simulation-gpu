#include <sqlite3.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

#include "qTensor.cuh"
#include "Contraction.hpp"

#ifdef ENABLE_CUDA
#include "qTensorCUDA.cuh"
#endif

using namespace std;

bool retrieveData(sqlite3* db, std::vector<Contraction>& contractions, int programId) {
    sqlite3_stmt* stmt;
    int rc;

    std::string query = "\
        SELECT c.id, c.program_id, c.span, c.left_id, c.right_id, c.kind, g.data  \
        FROM contractions as c left join gates as g on c.gate_id == g.id  \
        WHERE program_id = ?";
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, programId);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        Contraction contraction;
        contraction.id = sqlite3_column_int(stmt, 0);
        contraction.programId = sqlite3_column_int(stmt, 1);
        string span_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        span_str = span_str.substr(1, span_str.size() - 2);
        size_t pos = 0;

        // TODO : ------------------------------------------------------ fix implicitly tensor expanded gates
        auto span = std::vector<unsigned char>();
        while ((pos = span_str.find(',')) != string::npos) {
            span.push_back(stoi(span_str.substr(0, pos)));
            span_str.erase(0, pos + 1);
        }
        span.push_back(stoi(span_str));
        for (unsigned char i = 0; i < span.size(); i++) contraction.span.push_back(span[i]); 
        // auto start = span[0];
        // auto end = span[span.size() - 1];
        // for (unsigned char i = start; i <= end; i++) { contraction.span.push_back(i); printf("Pushing %d\n", i); }
        // TODO : ------------------------------------------------------ fix implicitly tensor expanded gates

        contraction.leftId = sqlite3_column_int(stmt, 3);
        contraction.rightId = sqlite3_column_int(stmt, 4);
        contraction.kind = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        if (sqlite3_column_type(stmt, 6) != SQLITE_NULL) {
            const void* data = sqlite3_column_blob(stmt, 6);
            size_t dataSize = sqlite3_column_bytes(stmt, 6);
            std::vector<double> values(static_cast<const double*>(data), static_cast<const double*>(data) + dataSize / sizeof(double));
            contraction.data = QTensor(values, contraction.span);
        }
        cudaStreamCreate(&contraction.stream);
        contractions.push_back(contraction);
    }

    sqlite3_finalize(stmt);

    for (auto& contraction : contractions) {
        if (contraction.kind == "C") {
            for (auto& c : contractions) {
                if (c.id == contraction.leftId) {
                    contraction.left = &c;
                }
                if (c.id == contraction.rightId) {
                    contraction.right = &c;
                }
            }
        }
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Error retrieving contractions: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Error retrieving gates: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    return true;
}

// Function to retrieve the unitary matrix from the database as a BLOB
bool getUnitaryMatrixFromDB(const char* dbName, int programId, std::complex<double>*& unitaryMatrix, int& matrixSize, int& qiskit_unitary_gpu_time, int& tree_building_time) {
    sqlite3* db;
    sqlite3_stmt* stmt;
    const char* query = "SELECT unitary_matrix, qiskit_unitary_gpu_time_ms, tree_building_time_ms FROM programs WHERE id = ?";
    
    // Open SQLite database
    if (sqlite3_open(dbName, &db) != SQLITE_OK) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return false;
    }

    // Prepare the SQL statement
    if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return false;
    }

    // Bind program ID
    if (sqlite3_bind_int(stmt, 1, programId) != SQLITE_OK) {
        std::cerr << "Failed to bind program ID: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return false;
    }

    // Execute query and fetch result
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* blobData = sqlite3_column_blob(stmt, 0);
        int blobSize = sqlite3_column_bytes(stmt, 0);
        qiskit_unitary_gpu_time = sqlite3_column_int(stmt, 1);
        tree_building_time = sqlite3_column_int(stmt, 2);

        if (blobData && blobSize > 0) {
            // Assuming the BLOB contains a square matrix of std::complex<dtype>
            matrixSize = sqrt(blobSize / sizeof(std::complex<double>));  // Calculate size assuming square matrix

            // Allocate memory for the unitary matrix
            unitaryMatrix = new std::complex<double>[matrixSize * matrixSize];

            // Copy BLOB data into unitaryMatrix
            memcpy(unitaryMatrix, blobData, blobSize);
        } else {
            std::cerr << "No unitary_matrix found or empty." << std::endl;
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            return false;
        }
    } else {
        std::cerr << "No matching row found." << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return false;
    }

    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return true;
}

// Function to compare two matrices element-wise
double compareMatrices(const std::complex<double>* mat1, const std::complex<dtype>* mat2, int size) {
    double norm = 0.0;
    for (int i = 0; i < size * size; ++i) {
        norm += std::norm(mat1[i] - (std::complex<double>)mat2[i]);
    }
    return sqrt(norm);
}

ostream& operator<<(ostream& os, const vector<unsigned char>& span) {
    os << "[";
    for (size_t i = 0; i < span.size(); ++i) {
        os << (size_t)span[i];
        if (i < span.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

void printTree(Contraction* root, size_t level = 0) {
    if (root == nullptr) {
        return;
    }

    std::cout << std::string(level * 2, ' ') << "Contraction: " << root->span << " (Kind: " << root->kind << ")" << std::endl;

    if (root->kind == "G") {
        std::cout << std::string((level + 1) * 2, ' ') << "Gate " << " rank: " << (root->data).getRank() << ")" << std::endl;
    } else {
        printTree(root->left, level + 1);
        printTree(root->right, level + 1);
    }
} 

#define DEBUG false

string db_name = "../data/qiskit.db";

int main(int argc, char** argv) {
    sqlite3* db;
    // int rc = sqlite3_open("../data/db_ours.sqlite", &db);
    int rc = sqlite3_open(db_name.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <program_id>" << std::endl;
        sqlite3_close(db);
        return 1;
    }

    #ifdef ENABLE_CUDA
    cout << "CUDA enabled" << endl;
    #else
    cout << "CUDA disabled" << endl;
    #endif

    std::vector<Contraction> contractions;
    // MOD
    int programId = std::stoi(argv[1]);

    if (retrieveData(db, contractions, programId)) {
        if (DEBUG){
        for (const auto& contraction : contractions) {
            std::cout << "-----------------------------------" << std::endl;
            std::cout << "Contraction: " << contraction.id << std::endl;
            std::cout << "Span: " << contraction.span << std::endl;
            std::cout << "Left ID: " << contraction.leftId << std::endl;
            std::cout << "Right ID: " << contraction.rightId << std::endl;
            std::cout << "Kind: " << contraction.kind << std::endl;
            std::cout << "Data rank: " << contraction.data.getRank() << std::endl;
            std::cout << "Data:\n";
            contraction.data.printValues();
            std::cout << std::endl;

        }
        }
    }

    cout << "-----------------------------------" << endl;
    cout << "Starting contraction" << endl;


    // root is last

    Contraction* root = contractions.empty() ? nullptr : &contractions.back();
    if (DEBUG) {
        printTree(root);
    }
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    #ifdef ENABLE_CUDA
    contractTreeGPU(root);
    #endif
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time = " << (double)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000. << "[ms]" << endl;
    // append the time and the program id to times.csv
    if (DEBUG){
        for (const auto& contraction : contractions) {
            std::cout << "Contraction: " << contraction.id << std::endl;
            std::cout << "Span: " << contraction.span << std::endl;
            std::cout << "Left ID: " << contraction.leftId << std::endl;
            std::cout << "Right ID: " << contraction.rightId << std::endl;
            std::cout << "Kind: " << contraction.kind << std::endl;
            std::cout << "Data rank: " << contraction.data.getRank() << std::endl;
            std::cout << "Data:\n"; 
            contraction.data.printValues();
            std::cout << std::endl;
        }
    }

    std::complex<double>* dbMatrix = nullptr;  // Matrix from database
    int dbMatrixSize, qiskit_unitary_gpu_time, tree_building_time;
    auto ok = getUnitaryMatrixFromDB(db_name.c_str(), programId, dbMatrix, dbMatrixSize, qiskit_unitary_gpu_time, tree_building_time);
    if (!ok) {
        cout << "Error getting unitary matrix from database" << endl;
        return 1;
    }
    auto error = compareMatrices(dbMatrix, (contractions.end()-1)->data.values, dbMatrixSize);
    cout << "Error: " << error << endl;
    if (!(error < 1e-3)) {
        cout << "Unitary matrix does not match the last contraction" << endl;
    }
    ofstream out("unitary.txt");
    (contractions.end()-1)->data.printValues(out);
    out.close();

    sqlite3_close(db);

    ofstream times("times.csv", ios::app);
    double milliseconds_elapsed = (double)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000.;
    times << milliseconds_elapsed << "," << qiskit_unitary_gpu_time << "," << (double)qiskit_unitary_gpu_time / (double)(milliseconds_elapsed + tree_building_time) << "," << tree_building_time  << "," << programId << endl;
    times.close();


    return 0;
}