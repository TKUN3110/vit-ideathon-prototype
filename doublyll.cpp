#include <iostream>
using namespace std;


struct Node {
    int val;
    Node* forward;
    Node* backward;
};


Node* buildNode(int item) {
    Node* freshNode = new Node;
    freshNode->val = item;
    freshNode->forward = NULL;
    freshNode->backward = NULL;
    return freshNode;
}


void pushFront(Node*& start, Node*& end, int item) {
    Node* freshNode = buildNode(item);
    if (start == NULL) {
        start = end = freshNode;
    } else {
        freshNode->forward = start;
        start->backward = freshNode;
        start = freshNode;
    }
}


void pushBack(Node*& start, Node*& end, int item) {
    Node* freshNode = buildNode(item);
    if (start == NULL) {
        start = end = freshNode;
    } else {
        freshNode->backward = end;
        end->forward = freshNode;
        end = freshNode;
    }
}


void pushAt(Node*& start, Node*& end, int item, int index) {
    if (index == 1) {
        pushFront(start, end, item);
        return;
    }
    Node* curr = start;
    for (int i = 1; i < index - 1 && curr != NULL; i++) {
        curr = curr->forward;
    }
    if (curr == NULL) {
        cout << "Invalid position" << endl;
        return;
    }
    if (curr == end) {
        pushBack(start, end, item);
        return;
    }
    Node* freshNode = buildNode(item);
    freshNode->forward = curr->forward;
    freshNode->backward = curr;
    curr->forward->backward = freshNode;
    curr->forward = freshNode;
}


void locate(Node* start, int target) {
    Node* curr = start;
    int idx = 1;
    while (curr != NULL) {
        if (curr->val == target) {
            cout << "Value found at position " << idx << endl;
            return;
        }
        curr = curr->forward;
        idx++;
    }
    cout << "Value not found" << endl;
}


void removeElement(Node*& start, Node*& end, int target) {
    Node* curr = start;
    while (curr != NULL && curr->val != target) {
        curr = curr->forward;
    }
    if (curr == NULL) {
        cout << "Value not found" << endl;
        return;
    }
    if (curr == start) {
        start = start->forward;
        if (start != NULL) {
            start->backward = NULL;
        } else {
            end = NULL;
        }
    } else if (curr == end) {
        end = end->backward;
        end->forward = NULL;
    } else {
        curr->backward->forward = curr->forward;
        curr->forward->backward = curr->backward;
    }
    delete curr;
    cout << "Value deleted successfully." << endl;
}


void printAscending(Node* start) {
    Node* curr = start;
    cout << "Forward: ";
    while (curr != NULL) {
        cout << curr->val << " ";
        curr = curr->forward;
    }
    cout << endl;
}


void printDescending(Node* end) {
    Node* curr = end;
    cout << "Backward: ";
    while (curr != NULL) {
        cout << curr->val << " ";
        curr = curr->backward;
    }
    cout << endl;
}


int main() {
    Node* first = NULL;
    Node* last = NULL;
    int option, element, location, total;
    do {
        cout << "\n1. Create" << endl;
        cout << "2. Insert at Beginning" << endl;
        cout << "3. Insert at End" << endl;
        cout << "4. Insert at Position" << endl;
        cout << "5. Search" << endl;
        cout << "6. Delete Value" << endl;
        cout << "7. Forward Traversal" << endl;
        cout << "8. Backward Traversal" << endl;
        cout << "9. Exit" << endl;
        cout << "Enter choice: ";
        cin >> option;
        switch (option) {
            case 1:
                cout << "Enter number of nodes: ";
                cin >> total;
                for (int i = 0; i < total; i++) {
                    cout << "Enter value: ";
                    cin >> element;
                    pushBack(first, last, element);
                }
                break;
            case 2:
                cout << "Enter value: ";
                cin >> element;
                pushFront(first, last, element);
                break;
            case 3:
                cout << "Enter value: ";
                cin >> element;
                pushBack(first, last, element);
                break;
            case 4:
                cout << "Enter value: ";
                cin >> element;
                cout << "Enter position: ";
                cin >> location;
                pushAt(first, last, element, location);
                break;
            case 5:
                cout << "Enter value to search: ";
                cin >> element;
                locate(first, element);
                break;
            case 6:
                cout << "Enter value to delete: ";
                cin >> element;
                removeElement(first, last, element);
                break;
            case 7:
                printAscending(first);
                break;
            case 8:
                printDescending(last);
                break;
            case 9:
                cout << "Exit" << endl;
                break;
            default:
                cout << "Invalid choice" << endl;
        }
    } while (option != 9);
    return 0;}
