// codemirror6/helpers/babel-doc.mjs

const functions = {};
export { functions };

function cleanDocComment(comment) {
    if (!comment) return '';
    // This regex removes the leading " * " or " *" from each line.
    return comment
        .split('\n')
        .map(line => line.replace(/^\s*\*\s?/, ''))
        .join('\n')
        .trim();
}

function buildDocumentation(funName, funPath, comment) {
    const funNode = funPath.node;
    const params = funNode.params.map(p => {
        if (p.type === "RestElement") return `...${p.argument.name}`;
        if (p.type === "AssignmentPattern") {
            try {
                const rightSide = funPath.get('params')[funNode.params.indexOf(p)].get('right').getSource();
                return `${p.left.name} = ${rightSide}`;
            } catch(e) {
                 return `${p.left.name} = ...`;
            }
        }
        if (p.type === "Identifier") return p.name;
        return '?';
    }).join(', ');

    const signature = `${funName}(${params})`;
    const cleanedComment = cleanDocComment(comment);
    
    return `${signature}\n${cleanedComment}`.trim();
}

export default function babelDocPlugin() {
    function processPath(funName, funPath, commentNode) {
        if (!funName) return;
        
        const leadingComments = commentNode.leadingComments;
        if (!leadingComments || leadingComments.length === 0) return;

        // Find the last block comment before the function
        const lastComment = leadingComments.filter(c => c.type === 'CommentBlock').pop();
        if (lastComment) {
            functions[funName] = buildDocumentation(funName, funPath, lastComment.value);
        }
    }

    return {
        name: 'babel-plugin-doc',
        visitor: {
            FunctionDeclaration(path) {
                processPath(path.node.id?.name, path, path.node);
            },
            AssignmentExpression(path) {
                const right = path.get('right');
                if (right.isFunctionExpression() || right.isArrowFunctionExpression()) {
                    const funName = path.get('left').getSource();
                    const statementNode = path.getStatementParent().node;
                    processPath(funName, right, statementNode);
                }
            },
        },
    };
}
